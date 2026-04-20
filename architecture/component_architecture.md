# Component Architecture

Shows the React frontend and FastAPI backend components and their interactions.

```mermaid
flowchart TD
    subgraph Frontend["Frontend - React :5173"]
        WC["Webcam.jsx"] --> VID["&lt;video&gt; ref"]
        VID --> CAN["&lt;canvas&gt; ref"]
        CAN --> API["apiPredict()"]
    end

    subgraph Backend["Backend - FastAPI :8000"]
        EP["POST /predict"] --> READ["_read_image()"]
        READ --> PRE["preprocess()"]
        PRE --> MOD["Keras Model"]
    end

    API -->|"JPEG blob"| EP
    MOD -->|"JSON response"| API
    API --> WC

    style Frontend fill:#0f1f0f,stroke:#22c55e,stroke-width:2px,color:#e0e0e0
    style Backend fill:#1f0f0f,stroke:#f97316,stroke-width:2px,color:#e0e0e0
```

**Flow:**
1. User starts camera in Webcam.jsx
2. Every 100-200ms, canvas captures video frame
3. apiPredict() sends JPEG blob to backend
4. Backend processes and returns prediction
5. UI updates with results
