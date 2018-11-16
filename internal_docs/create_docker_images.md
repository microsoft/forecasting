## Guideline for Creating Docker Images

We recommend that the submitter provides a Docker image for the reproducibility of the submission. Docker is a platform for developers to develop, deploy,
and run applications with containers. Please refer to [this tutorial](https://docs.docker.com/get-started/) to learn basics of Docker. Here we introduce the 
steps of creating and publishing the Docker image.

A new Docker image can be created by modifying the Dockerfile of the baseline implementation of a given benchmark. For instance, there is a Dockerfile in 
`retail_sales/OrangeJuice_Pt_3Weeks_Weekly/baseline/Naive` folder used in the baseline model of retail sales forecasting. In the beginning of this Dockerfile 
we load [ubuntu:16.04](https://hub.docker.com/_/ubuntu/) image as the base image. Then, we install a list of basic Linux packages which are required 
for installing other packages or needed in the model development. Afterwards, we install an R environment with r-base version 3.5.1. Finally, we install R 
dependencies with the following commands
```bash
RUN echo 'options(repos = list(CRAN = "http://mran.revolutionanalytics.com/snapshot/2018-08-27/"))' >> /etc/R/Rprofile.site
ADD ./install_R_dependencies.r /tmp
RUN Rscript install_R_dependencies.r
```
where `install_R_dependencies.r` is an R script that specifies and installs a list of R packages (See the R dependency file in `retail_sales/OrangeJuice_Pt_3Weeks_Weekly/baseline/Naive` folder as an example). You can modify the listed packages based on your 
need. To ensure the same R package version is installed, we use a MRAN snapshot URL to download packages archived on a specific date which can also be 
customized. In case you need to install Python packages, we suggest you first update your `pip` via adding the following command to the Dockerfile
```bash
RUN pip3 install --upgrade pip
```
if you use Python 3. Then, you can mount a Python dependency file into the Docker container and install Python dependencies using the following commands
```bash
WORKDIR /tmp
ADD ./python_dependencies.txt /tmp
RUN pip3 install -r python_dependencies.txt
```
where `python_dependencies.txt` is a file specifying the Python packages and versions (See the Python dependency file in 
`retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM` folder as an example). Again you can update the listed packages there if necessary. Note that you will need to use `RUN pip` command if you are working with Python 2.

After customizing the Dockerfile and dependency files, you can build a local Docker image in a Linux VM by following the steps below:

1. Make sure Docker is installed. You can check if Docker is installed in your VM by running
    ```bash
    docker -v
    ```
    You will see the Docker version if Docker is installed. If not, you can install it by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Note that if you want to execute Docker commands as a non-root user, you need to create a Unix group and add users to it by following the instructions [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user). Otherwise, you need to run the commands with sudo. 

2. Build Docker image by running
    ```bash
    docker build -t <image name> .
    ```
    from the submission folder where the Dockerfile and dependency files reside. Here `<image name>` is the name of the local Docker image. An example name  is `lightgbm_image:v1`, where `v1` indicates the version of the Docker image. It may take tens of minutes to build the Docker image for the first time. But the process could be much faster if you rebuild the image after applying small changes to the Dockerfile or dependency files, since previous Docker building steps will be cached and most of them will not be repeated.  
    
3. After the Docker image is built, you may need to test your model training and scoring script inside a Docker container created from this image. To do this, you will need to
    * 3.1 Choose a name for a new Docker container and create it by running the following command from `/TSPerf` folder (assuming that you've cloned TSPerf repository):
        ```bash
        docker run -it -v $(pwd):/TSPerf --name <container name> <image name>
        ```
        Note that option `-v $(pwd):/TSPerf` allows you to mount `/TSPerf` folder (the one you cloned) to the container so that you will have access to the source code and data in the container. Here `<container name>` is the name of the Docker container, e.g. `lightgbm_container`. You will automatically enter the Docker container after executing the above command. 
    * 3.2 Inside the Docker container, train the model and make predictions by running the following command from `/TSPerf` folder
        ```bash
        source ./common/train_score_vm <submission path> <script type> 
        ```
        where `train_score_vm` is a bash script that invokes the model training and scoring script; `<submission path>` and `<script type>` are the path of the submission folder (e.g., `./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM`) and type of the script (R, Python, or Python3), respectively. 

4. If the above test goes smoothly, we can push the Docker image to the Azure Container Registry (ACR) with the following steps:
    * 4.1 Log into Azure Container Registry (ACR)
    ```bash
    docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
    ``` 
    where `<ACR Acccess Key>` can be found [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/accessKey).
    * 4.2 Create a tag that refers to the Docker image
    ```bash
    docker tag <image name> <tag name>
    ```
    where `<tag name>` is the name of the Docker image in the ACR. We recommend to name the tag using the convention `tsperf.azurecr.io/<benchmark directory>/<image name>`, e.g. `tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/lightgbm_image:v1`.
    * 4.3 Push the Docker image to ACR
    ```bash
    docker push <tag name>
    ```
    with `<tag name>` being the one that you picked in the last step. Make sure that your tag name starts with `tsperf.azurecr.io/`, otherwise the docker push command will not work. You can find the Docker image [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/repository), after it is successfully pushed to ACR.