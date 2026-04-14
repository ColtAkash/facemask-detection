// Thin client for the FastAPI backend (see ../../../backend/app.py).
// Configure the base URL via VITE_API_URL in .env; defaults to localhost:8000.

export const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export async function apiHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`health ${res.status}`)
  return res.json()
}

// Upload a Blob/File as multipart/form-data to a POST endpoint.
async function postBlob(path, blob) {
  const fd = new FormData()
  fd.append('file', blob, 'frame.jpg')
  const res = await fetch(`${API_BASE}${path}`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`${path} ${res.status}`)
  return res.json()
}

export const apiPredict       = (blob) => postBlob('/predict', blob)
export const apiPredictFaces  = (blob) => postBlob('/predict/faces', blob)

// Classes are returned by the server; keep a stable index order matching CLASS_INFO in the UI.
export const LABEL_INDEX = {
  with_mask: 0,
  without_mask: 1,
  mask_weared_incorrect: 2,
}
