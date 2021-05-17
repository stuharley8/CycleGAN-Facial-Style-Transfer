# <center>Applications of Generative adversarial Networks for Facial Style Transfer</center>
### <center>Kyle Rodrigues, Nathan Chapman, Nathan DuPont, Kiel Dowdle, Stuart Harley</center>
### <center>{rodriguesk, chapmann, dupontn, dowdlek, harleys}@msoe.edu</center>

---

## <center>Project Abstract</center>

The evolution of AI (Artificial Intelligence) technologies has advanced the development of human society and is transforming every walk of life. With the importance of this technology in our lives, Milwaukee School of Engineering (MSOE) is collaborating with Discovery World and Rockwell Automation to develop a new interactive exhibit at Discovery World that provides an educational opportunity for AI that is not currently available to this extent in the Milwaukee area. Developing an interactive exhibit based around AI is a challenging task to accomplish for many reasons; one of the largest reasons being the diversity and age present within the target population. To provide all patrons with the best experience, a solution needs to be made such that it will not express any forms of bias towards the attendees, allowing for all individuals to get engaged and find potential interest in AI. With those considerations in mind, we developed a web-based application capable of detecting a face from an image, which then uses a Generative Adversarial Network (GAN) to apply a style transfer in real time. This web application is based on open-source code (OpenCV and CycleGAN) and could eventually be  extended and deployed at Discovery World. 

[View our research paper here](./resources/cs3310_group01_final-project.pdf)
<br><br>

## <center>Desired Architecture</center>

This project has been created with the intention of being packaged within a Docker container before production use. Development to the project can be done through this repository, and once completed, [TODO] can be run to generate a Docker Image, and push it to Docker Hub. 

This image can then be obtained with the target exhibit through pulling the latest image (or any other desired tagged image), and running the container to start the application.

![Production Architecture for Application Deployment](./imgs/dw-gan-prod-architecture.png)

<br><br>

## <center>Running the Application</center>

Docker images of this project are publicly hosted on Docker Hub. You can access the public images using the commands below, with support for both CPU and GPU platforms.

**CPU Version**
```
docker pull nathandupon/dw-gan:cpu
```

**GPU Version**
```
docker pull nathandupon/dw-gan:gpu
```
***Note***: This version requires hardware capable of running CUDA with NVIDIA drivers. If you are unsure if your hardware can run this image, please view the [CUDA base image](https://hub.docker.com/r/nvidia/cuda) for more information.

<br><br>

## <center>Configuring Local Development</center>

In order to develop locally with this project, you will likely need the following dependencies installed on your system:
- [Docker](https://docs.docker.com/get-docker/)
- [Anaconda](https://www.anaconda.com/products/individual)

This repository consists of everything required to build the application from source, as well as build and publish the application within a Docker container. 

### Developing From Source

To build from source, follow these steps:

1. From within the `/CycleGAN-Server/` directory, run the following command (with windows or linux selected based on your host operating system):
```
conda env create -f environment-[windows | linux].yml
```
2. Run the following command from within the `/CycleGAN-Server/flask` directory to install local pip packages:
```
cd ./flask/packages/cyclegan && pip install -e .
```
3. Run the following command within the `/CycleGAN-Server/flask` to start the application. Add the optional `--gpu_ids` flag if you wish to run your program with a supported GPU, otherwise it will run on CPU by default:
```
python app.py [--gpu_ids=0]
```

### Developing From Docker

Dockerfiles are included for both CPU and GPU local development. To build either container, run the following commands (with CPU or GPU specified):

```
docker build -f Dockerfile-[cpu | gpu] --tag dw-gan .
docker run -p 5000:5000 --name dw-gan dw-gan
```

This will produce a Docker image running on your host. The application should be viewable at [http://localhost:5000](http://localhost:5000).

In order to publish this container for use remotely, run the following commands:

```
docker build -f Dockerfile-[cpu | gpu] --tag [DockerHub Repository]/dw-gan:[cpu | gpu] .
docker push [DockerHub Repository]/dw-gan:[cpu | gpu]
```

These steps are outlined in the Docker documentation (with additional information), which can be [found here](https://docs.docker.com/docker-hub/).


