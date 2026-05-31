# Image Similarity Search

## Installation

### 1. **Install dependencies:**
1.1. Create and activate a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

1.2. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

### 2. **Download data-images** 
2.1. Move data-images folder from https://github.com/msikyna/MetricLearningExperiment2 to the root directory of the project.

2.2. Into the data-images folder, add the 561.txt file from https://drive.google.com/drive/folders/1sFfDmAFW5OcI3wNIEdYVQsZnUoxp-N-5?usp=sharing directly into the data-images folder.

2.3. The final structure should look like this:
   ```
   /sim-search
   ├── data-images
   │   ├── 561
   │   │    ├── 000.jpg
   │   │    ├── 001.jpg
   │   │    ├── 002.jpg
   │   │    └── ...
   │   └── 561.txt
   ├── requirements.txt
   ├── README.md
   ├── manage.py 
   ├── djangoProject
   ├── base
   └── ...
   ```
### 3. **Run the server:**
##### 1.
```bash
python manage.py makemigrations
```
##### 2.
```bash
python manage.py migrate
```

### In Bash:
#####
   ```bash
   python manage.py runserver
   ```
### OR
### In PyCharm:

1. **Open Run Configuration Settings:**
   - Click the "Add Configuration" button in the top-right toolbar
   - Or go to `Run` → `Edit Configurations...`

2. **Add Django Server Configuration:**
   - Click the `+` button
   - Select `Django Server` from the list

3. **Configure Settings:**
   - **Name:** Django Server
   - **Host:** `127.0.0.1`
   - **Port:** `8000`
   - **Python interpreter:** Ensure your project's Python interpreter is selected
   - **Working directory:** Should point to your project root (where `manage.py` is located)
   - **Environment variables:** Leave blank unless needed

4. **Apply and Run:**
   - Click `Apply` and `OK`
   - Click the green play button (▶) in the toolbar to start the server


### 4. **Access the application:**
Open your web browser and navigate to `http://127.0.0.1:8000/` to use the image similarity search application.

## Usage
1. Select a base image from selection or search by text

## Docker Deployment (Server)

This setup is prepared so you can pull the repo and run with Docker directly.

### 1. Prepare environment
```bash
cp .env.example .env
mkdir -p runtime user_matrices cache/L2 cache/IP
```

### 2. Build and run
```bash
docker compose up -d --build
```

If the project was moved and the new `./cache` bind mount is empty or
root-owned, seed FAISS mapping files from the old deployment cache:

```bash
SIMSEARCH_SEED_CACHE_ROOT=/home/xsikyna/sim-search/cache \
  docker compose -f docker-compose.yml -f docker-compose.seed-cache.yml up -d --build
```

### 3. Check logs
```bash
docker compose logs -f sim-search
```

### 4. Update after code changes
```bash
git pull
docker compose up -d --build
```

### Final URL target
The app is configured for URL prefix:
```text
http://disa.fi.muni.cz/demos/personalized-similarity-search
```
through:
```bash
DJANGO_FORCE_SCRIPT_NAME=/demos/personalized-similarity-search
```

### Reverse proxy (Apache) example
```apache
ProxyPreserveHost On
ProxyPass        /demos/personalized-similarity-search/  http://cybela14.fi.muni.cz:8942/
ProxyPassReverse /demos/personalized-similarity-search/  http://cybela14.fi.muni.cz:8942/
RequestHeader set X-Forwarded-Proto "https"
```

### Container/network notes
- Gunicorn inside container binds `0.0.0.0:8942`.
- Published port is `${SIMSEARCH_BIND_IP}:${SIMSEARCH_PORT}` (default `0.0.0.0:8942`).

### Concurrency tuning for ~20 simultaneous users
The repo defaults are tuned for a modest multi-user deployment:

- `GUNICORN_WORKERS=8`
- `FAISS_NUM_THREADS=1`
- file-based Django sessions in `/app/runtime/sessions`
- SQLite WAL + busy timeout enabled
- `SIMSEARCH_QUERY_LOG_EXPAND_RESULTS=0` to avoid a second search only for logging
- in-process user matrix cache enabled (`SIMSEARCH_MATRIX_CACHE_SIZE=32`)

Recommended next step on the server:

1. Keep only the index you really use enabled.
2. Serve `/images/...` outside Gunicorn if possible.

Why step 2 matters:
- the search backend can handle concurrency reasonably with multiple workers
- image streaming from shared storage through Gunicorn will still consume workers quickly

If your reverse proxy supports it, prefer one of these:

- Apache/Nginx direct file serving for `/images/`
- or enable `SIMSEARCH_IMAGE_X_SENDFILE=1` / `SIMSEARCH_IMAGE_X_ACCEL_REDIRECT_PREFIX=...`

That change usually helps concurrency more than just increasing Gunicorn workers.

## Load Testing

There is a standalone HTTP load test script at:

```bash
python scripts/loadtest_sim_search.py --help
```

It exercises the real personalization flow over HTTP:

1. login
2. text search
3. save positive and negative feedback
4. apply feedback to learn/update the matrix
5. rerun the query
6. optionally fetch the progressive full rerun

### Example: 20 concurrent users

```bash
python scripts/loadtest_sim_search.py \
  --base-url http://disa.fi.muni.cz/demos/personalized-similarity-search/ \
  --users 20 \
  --rounds 5 \
  --num-results 20 \
  --metric cosine \
  --output-json runtime/loadtest_20_users.json
```

Notes:

- By default the script prepares test users named `loadtest_user_001`, `loadtest_user_002`, ...
- The first version uses the text-search path because it covers the full search + feedback + matrix-learning flow without needing extra image-anchor fixtures.
- If you want to reuse pre-created users, add `--no-prepare-users`.
