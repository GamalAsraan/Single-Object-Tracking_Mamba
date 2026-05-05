import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        # Phase 9c: best val loss tracking (persisted across resumes)
        self.best_val_loss = float('inf')

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        Phase 9c patches:
          - KeyboardInterrupt (cell stop) → save checkpoint + exit cleanly
          - bare except: → except Exception: (so KeyboardInterrupt isn't swallowed)
          - best-model saving after val epochs
        """
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)

                    # ── Periodic checkpoint (existing logic) ──
                    save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                    save_epochs = [79, 159, 239]
                    if epoch > (max_epochs - 1) or save_every_epoch or epoch % 5 == 0 or epoch in save_epochs or epoch > (max_epochs - 5):
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()

                    # ── Phase 9c: best-model check after val epochs ──
                    self._check_save_best()

            except KeyboardInterrupt:
                # ── Phase 9c: graceful stop on cell interrupt ──
                print('\n' + '=' * 70)
                print('⚠️  Training stopped by user at epoch {}'.format(self.epoch))
                print('=' * 70)
                if self._checkpoint_dir and self.settings.local_rank in [-1, 0]:
                    self.save_checkpoint()
                    print('✅ Checkpoint saved at epoch {}'.format(self.epoch))
                    best_str = '  (best val loss so far: {:.5f})'.format(self.best_val_loss) \
                        if self.best_val_loss < float('inf') else ''
                    print('   Resume by re-running the launcher cell.{}'.format(best_str))
                print('=' * 70)
                return  # exit cleanly — don't restart

            except Exception:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def _check_save_best(self):
        """Phase 9c: compare last val loss to best and save if improved."""
        val_loss = getattr(self, '_last_val_loss', None)
        if val_loss is None:
            return  # no val ran this epoch

        if val_loss < self.best_val_loss:
            prev = self.best_val_loss
            self.best_val_loss = val_loss
            if self._checkpoint_dir and self.settings.local_rank in [-1, 0]:
                self._save_best_checkpoint(val_loss)
                if prev < float('inf'):
                    print('🏆 New best val loss: {:.5f} (prev {:.5f})'.format(val_loss, prev))
                else:
                    print('🏆 First best val loss: {:.5f}'.format(val_loss))
        else:
            print('   Val loss {:.5f} did not improve from best {:.5f}'.format(val_loss, self.best_val_loss))

        # Reset so we don't re-check on non-val epochs
        self._last_val_loss = None

    def _save_best_checkpoint(self, val_loss):
        """Save a best-model checkpoint (overwrites previous best)."""
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        state = {
            'epoch': self.epoch,
            'actor_type': type(self.actor).__name__,
            'net_type': type(net).__name__,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings,
            'best_val_loss': val_loss,
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        tmp_path = os.path.join(directory, 'TrackingMamba_best.tmp')
        best_path = os.path.join(directory, 'TrackingMamba_best.pth.tar')
        torch.save(state, tmp_path)
        os.rename(tmp_path, best_path)
        print('   Saved best model → {}'.format(best_path))

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings,
            'best_val_loss': self.best_val_loss,   # Phase 9c: persist across resumes
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # weights_only=False because checkpoints contain pickled settings,
        # constructor, and optimizer state. (Patched for Phase 9b resume.)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Phase 9c: restore best_val_loss from checkpoint
        if 'best_val_loss' in checkpoint_dict:
            self.best_val_loss = checkpoint_dict['best_val_loss']
            print('Restored best_val_loss = {:.5f} from checkpoint'.format(self.best_val_loss))

        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)

        print('Loaded checkpoint: {} (epoch {})'.format(checkpoint_path, self.epoch))
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        print("Loading pretrained model from ", checkpoint_path)
        # weights_only=False: same rationale as load_checkpoint above.
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True