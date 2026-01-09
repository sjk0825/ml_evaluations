
### Create container
```
$ docker run -it --name evaluation_server_2 -p 5002:5000 -v ./:/workspace/ my_ubuntu:22.04 bash
```

### Start container
```
$ docker start evaluation_server_2
$ docker exec -it evaluation_server_2 /bin/bash
```

### Install conda
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ source ~/.bashrc
$ conda config --set auto_activate_base false
$ conda init
$ conda --version
```

### set environment
```
$ export OPENAI_API_KEY="sk-proj-xxxxxx"
$ source ~/.bashrc
```