```env
FLASK_APP=run_api.py
FLASK_ENV=development
PORT=5000

# MongoDB Atlas Settings
# Replace <username>, <password>, and <cluster-url> with your Atlas details
MONGO_URI=
MONGO_DB_NAME=odp_rag_db
MONGO_COLLECTION_NAME=video_nodes

# Redis Settings (Message Broker for Celery)
# If running Redis locally, this is the default URI
REDIS_URI=redis://localhost:6379/0
```
