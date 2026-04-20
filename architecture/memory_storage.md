# Memory Storage Flow

Shows how image data flows through RAM only - no disk writes at any point.

```mermaid
flowchart LR
    subgraph Browser["Browser - RAM Only"]
        direction LR
        V["Video Stream<br/>640×480"] --> C["Canvas<br/>320×320"] --> B["JPEG Blob<br/>~8-15 KB"]
    end

    subgraph Network["HTTP Request"]
        direction LR
        R["POST /predict<br/>multipart/form-data<br/>binary JPEG data"]
    end

    subgraph Backend["Backend - RAM Only"]
        direction LR
        U["UploadFile<br/>bytes"] --> P["PIL Image<br/>RGB 320×320"] --> N["NumPy Array<br/>224×224×3"]
    end

    B --> R --> U

    style Browser fill:#0f1f0f,stroke:#22c55e,stroke-width:2px,color:#e0e0e0
    style Network fill:#0f0f1f,stroke:#3b82f6,stroke-width:2px,color:#e0e0e0
    style Backend fill:#1f0f0f,stroke:#f97316,stroke-width:2px,color:#e0e0e0
```

**Key Points:**
- All data remains in memory at every stage
- No temporary files written to disk
- Browser garbage collects Blob after request
- Backend garbage collects PIL/NumPy objects after response
