## 📌  Demo Deployments

**DockerFile**

- The Dockerfile (under the directory "src/docker_demo") has been configured to use GRADIO SERVER with PORT 8080. 
- This will enable to run the model as a web application on the GRADIO app (Over PORT 8080).

- Command to build the Docker image

```
cd src/docker_demo
```

```
docker image build -t emlov2-session-04 .
```

- Once the image is built, it can be run locally using the command

```
docker run -t -p 8080:8080 emlov2-session-04
```

**Pushing the DockerFile to Dockerhub**

- The image can be pushed to DockerHub and can be accessed directly from any remote server to run the predictions as web application.

```
docker login
```

```
docker tag emlov2-session-04 yelipetamanjunath/emlo2_a4
docker push yelipetamanjunath/emlo2_a4
```

- To run the docker image and access the model as web app, run the below command

```
docker run -t -p 8080:8080 yelipetamanjunath/emlo2_a4
```

- URL for the Docker image on DockerHub

https://hub.docker.com/repository/docker/yelipetamanjunath/emlo2_a4
