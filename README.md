# gh-test

Testing how to deploy a model to Cloud Run.

Everything is in the `Dockerfile` and `.github/workflows/deploy.yml`. 
The `Dockerfile` will download the pretrained weights from Google Cloud and add 
it to the image. The github actions take care of building the image, pushing it
to Google Registry Artifact and then deploying it to Cloud Run.

# Prerequisite

1. A working service account with the following permissions:
- `roles/run.admin`
- `roles/iam.serviceAccountUser`     (to act as the Cloud Run runtime service account)
- `roles/storage.admin`   (if using Google Container Registry (gcr) instead)
- `roles/artifactregistry.admin`     (project or repository level)
- `roles/iam.serviceAccountTokenCreator`  (to create the tokens for artifcat registry)

2. A secret in Github to leverage the service account
- `secrets.GCP_CREDENTIALS`