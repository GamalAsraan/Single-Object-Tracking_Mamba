# TrackingMamba Architecture Graph

The following Mermaid graph visualizes the end-to-end forward pass of the `TrackingMamba` model, from the raw template and search images down to the final bounding box heatmap and size predictions.

```mermaid
graph TD
    %% Define Styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef backbone fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef head fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:2px;

    %% 1. Inputs
    subgraph Inputs ["Input Images"]
        Z["Template Image (Z)<br>3 x Hz x Wz"]:::input
        X["Search Image (X)<br>3 x Hx x Wx"]:::input
    end

    %% 2. Tokenization Pipeline
    subgraph Tokenization ["Tokenization & Embedding"]
        PZ["PatchEmbed (Conv2D)<br>stride=16"]:::process
        PX["PatchEmbed (Conv2D)<br>stride=16"]:::process
        
        Z --> PZ
        X --> PX
        
        ZT["Template Tokens<br>Nz x D"]
        XT["Search Tokens<br>Nx x D"]
        
        PZ --> ZT
        PX --> XT
        
        PosZ["+ Add Absolute Positional Embedding"]:::process
        PosX["+ Add Absolute Positional Embedding"]:::process
        
        ZT --> PosZ
        XT --> PosX
    end

    %% 3. Sequence Concatenation
    subgraph Sequence ["Sequence Generation"]
        Concat["Concatenate Sequence<br>Shape: (Nz + Nx) x D"]:::process
        PosZ --> Concat
        PosX --> Concat
    end

    %% 4. VisionMamba Backbone
    subgraph Backbone ["VisionMamba Backbone (Depth: N)"]
        Mamba1["Mamba Block 1<br>(RMSNorm -> Mamba Mixer -> Add)"]:::backbone
        MambaDots[". . . . . ."]:::backbone
        MambaN["Mamba Block N<br>(RMSNorm -> Mamba Mixer -> Add)"]:::backbone
        
        Concat --> Mamba1
        Mamba1 --> MambaDots
        MambaDots --> MambaN
    end

    %% 5. Feature Extraction
    subgraph Reshape ["Feature Extraction"]
        Extract["Extract Search Tokens<br>(Take the last Nx tokens)"]:::process
        Unflatten["Reshape to Spatial 2D Grid<br>Shape: D x sqrt(Nx) x sqrt(Nx)"]:::process
        
        MambaN --> Extract
        Extract --> Unflatten
    end

    %% 6. Detection Head
    subgraph BoxHead ["Box Predictor Head"]
        CenterPred["Center Predictor<br>(5-layer Conv2D Stack)"]:::head
        
        Unflatten --> CenterPred
        
        Heatmap("Score Map (Center Heatmap)"):::output
        SizeMap("Size Map (Width, Height)"):::output
        OffsetMap("Offset Map (Sub-pixel shift)"):::output
        
        CenterPred --> Heatmap
        CenterPred --> SizeMap
        CenterPred --> OffsetMap
    end
```

## Architecture Details

1. **Tokenization:** Uses standard Patch Embedding (like ViT) to map spatial patches to continuous vectors of size `D` (e.g. 192). Positional encodings ensure the model understands where patches came from.
2. **Concatenation:** Crucially, the template tokens are prepended to the search tokens. Because the SSM is unidirectional, this ensures that the state encapsulates the appearance of the template target *before* scanning the search image.
3. **Mamba Backbone:** The sequence flows through stacked Mamba blocks where local contexts and long-range dependencies are integrated efficiently using parallel hardware scans.
4. **Reshaping:** Once sequence processing is complete, only the tokens corresponding to the search area are retained and folded back into a 2D spatial feature map.
5. **Box Head:** A parallelized convolutional head acts on the 2D feature map to predict the heatmaps from which the final bounding box coordinates are derived.
