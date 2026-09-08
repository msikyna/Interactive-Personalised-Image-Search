# Mahalanobis Search API

`POST /api/mahalanobis-search/` searches with a caller-supplied metric matrix. The endpoint is read-only and does not require a login or CSRF token.

With the production URL prefix, the path is `/demos/personalized-similarity-search/api/mahalanobis-search/`.

Send a JSON object containing `metric_matrix` and exactly one of:

- `query_text`: text to encode with CLIP.
- `query_image_index`: the numeric dataset index of an existing image. Images are addressed by dataset index; image blobs are not stored in the database.

The shorter aliases `matrix`, `query`, and `image_index` are also accepted.

The matrix must match the dataset embedding dimension (normally 768×768), be symmetric and positive semidefinite, and contain only finite numbers.

The response contains the query's full `embedding` (also exposed as top-level `query_embedding`) and up to 100 images ordered by ascending Mahalanobis distance. Every result includes `position` (and its `rank` alias), `index`, `image_name`, `image_url`, `distance`, and the full `embedding` vector.

## Text-query example

```python
import json
from urllib.request import Request, urlopen

import numpy as np

payload = {
    "query_text": "red sports car",
    "metric_matrix": np.eye(768).tolist(),
}
request = Request(
    "http://127.0.0.1:8000/api/mahalanobis-search/",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    print(json.load(response))
```

For an image query, use the same payload with `query_image_index` instead of `query_text`:

```json
{
  "query_image_index": 12345,
  "metric_matrix": [[1.0, 0.0], [0.0, 1.0]]
}
```

The 2×2 matrix above only illustrates the JSON shape; the deployed CLIP dataset normally requires 768 rows and columns. A saved payload can be sent with:

```bash
curl -X POST http://127.0.0.1:8000/api/mahalanobis-search/ \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

An abridged successful response looks like:

```json
{
  "success": true,
  "query": {
    "type": "text",
    "text": "red sports car",
    "embedding": [0.04, -0.01, 0.02]
  },
  "query_embedding": [0.04, -0.01, 0.02],
  "ordered_by": "distance_ascending",
  "search_mode": "faiss_candidate_rerank",
  "requested_result_count": 100,
  "result_count": 100,
  "results": [
    {
      "rank": 1,
      "position": 1,
      "index": 42,
      "image_name": "561/042.jpg",
      "image_url": "http://127.0.0.1:8000/images/561/042.jpg",
      "distance": 0.123,
      "embedding": [0.01, -0.02, 0.03]
    }
  ]
}
```

With FAISS enabled, the service uses FAISS candidates and orders those candidates by the supplied matrix; `search_mode` is `faiss_candidate_rerank`. With the in-memory backend, it scans the complete dataset and reports `exact_full_scan`. The default request limit is 32 MiB and can be changed with `DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE`.
