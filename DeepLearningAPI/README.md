# API for Deep Learning Model

## Overview

This API based on FastAPI is used to serve the model. It is deployed on Heroku and can be accessed using the
following command from root

```bash
uvicorn DeepLearningAPI.api:app --reload
```

The API will be available at http://localhost:8000/