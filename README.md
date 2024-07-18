# We <3 consuming cocktails

Add your Google API key to .env file like in .env.example (there's 2 of them and I am not sure which one is being used so add it to both #TeamMLway 😅)

Build docker image and run docker container:

```bash
docker build -t cocktailr-backend:latest .
docker run -p 8080:8080 -it cocktailr-backend:latest
```

Then you can call the API like:
```bash
curl --request POST \
  --url http://127.0.0.1:8080/ask-agent \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/9.3.2' \
  --data '{
	"question": "I have vodka, orange juice, and limes. Which cocktail can I make?"
}'
```


To deploy it on GCP (example, some of these should be environment variables):
```bash
gcloud init
gcloud artifacts repositories create quickstart-docker-repo --repository-format=docker --location=us-west2 --description="Docker repository"
gcloud builds submit --region=us-west2 --tag us-west2-docker.pkg.dev/gen-lang-client-0827333133/quickstart-docker-repo/cocktailr-backend:latest
gcloud run deploy cocktailr-backend --image us-west2-docker.pkg.dev/gen-lang-client-0827333133/quickstart-docker-repo/cocktailr-backend --region us-west2
```