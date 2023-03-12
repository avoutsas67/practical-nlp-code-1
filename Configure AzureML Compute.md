# Configure AzureML Compute

## Update libraries

``` bash
sudo apt-get update
sudo apt-get upgrade
sudo apt install software-properties-common
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
sudo apt update
sudo apt upgrade
sudo apt autoclean
sudo apt autoremove
```

## Mount Azure Storage Account - File share

[Mount SMB Azure file share on Linux](https://docs.microsoft.com/en-us/azure/storage/files/storage-how-to-use-files-linux?tabs=smb311)

### Prerequisites

#### Install dependencies

``` bash
sudo apt update
sudo apt install cifs-utils
```

#### Setup environment variables

``` bash
mount="/mnt"
resourceGroupName="natural-language-processing-01-rg"
storageAccountName="nlpcommon01sa"
fileShareName="development"
```

#### Setup credentials

``` bash
# Login to Azure
az login --use-device-code

# Get Storage Account Key
httpEndpoint=$(az storage account show \
    --resource-group $resourceGroupName \
    --name $storageAccountName \
    --query "primaryEndpoints.file" --output tsv | tr -d '"')
smbPath=$(echo $httpEndpoint | cut -c7-$(expr length $httpEndpoint))$fileShareName

# Get the storage account key for the indicated storage account.
# You must be logged in with az login and your user identity must have 
# permissions to list the storage account keys for this command to work.
storageAccountKey=$(az storage account keys list \
    --resource-group $resourceGroupName \
    --account-name $storageAccountName \
    --query "[0].value" --output tsv | tr -d '"')

sudo mount -t cifs $smbPath $mntPath -o username=$storageAccountName,password=$storageAccountKey,serverino
```

``` bash
# Create a folder to store the credentials for this storage account and
# any other that you might set up.
credentialRoot="/etc/smbcredentials"
sudo mkdir -p "/etc/smbcredentials"

# Create the credential file for this individual storage account
smbCredentialFile="$credentialRoot/$storageAccountName.cred"
if [ ! -f $smbCredentialFile ]; then
    echo "username=$storageAccountName" | sudo tee $smbCredentialFile > /dev/null
    echo "password=$storageAccountKey" | sudo tee -a $smbCredentialFile > /dev/null
else 
    echo "The credential file $smbCredentialFile already exists, and was not modified."
fi

# Change permissions on the credential file so only root can read or modify the password file.
sudo chmod 600 $smbCredentialFile
```

#### Static mount /etc/fstab

``` bash
httpEndpoint=$(az storage account show \
    --resource-group $resourceGroupName \
    --name $storageAccountName \
    --query "primaryEndpoints.file" --output tsv | tr -d '"')
smbPath=$(echo $httpEndpoint | cut -c7-$(expr length $httpEndpoint))$fileShareName

if [ -z "$(grep $smbPath\ $mntPath /etc/fstab)" ]; then
    echo "$smbPath $mntPath cifs nofail,credentials=$smbCredentialFile,serverino" | sudo tee -a /etc/fstab > /dev/null
else
    echo "/etc/fstab was not modified to avoid conflicting entries as this Azure file share was already present. You may want to double check /etc/fstab to ensure the configuration is as desired."
fi

sudo mount -a
```


##########

``` bash
sudo mkdir /mnt/development
if [ ! -d "/etc/smbcredentials" ]; then
sudo mkdir /etc/smbcredentials
fi
if [ ! -f "/etc/smbcredentials/nlpcommon01sa.cred" ]; then
    sudo bash -c 'echo "username=nlpcommon01sa" >> /etc/smbcredentials/nlpcommon01sa.cred'
    sudo bash -c 'echo "password=bOecJWiL951DIxehnEAToLhGnPhJy8r+aFGz2h0BNoBSxpvSAg8B8lAOn7j39UTENTkeWfhh8hjzXzl44ycZVQ==" >> /etc/smbcredentials/nlpcommon01sa.cred'
fi
sudo chmod 600 /etc/smbcredentials/nlpcommon01sa.cred

sudo bash -c 'echo "//nlpcommon01sa.file.core.windows.net/development /mount/nlpcommon01sa/development cifs nofail,vers=3.0,credentials=/etc/smbcredentials/nlpcommon01sa.cred,dir_mode=0777,file_mode=0777,gid=azureuser,serverino" >> /etc/fstab'
sudo mount -t cifs //nlpcommon01sa.file.core.windows.net/development /mount/nlpcommon01sa/development -o vers=3.0,credentials=/etc/smbcredentials/nlpcommon01sa.cred,dir_mode=0777,file_mode=0777,gid=azureuser,serverino
```

## Init conda

``` bash
conda init
```

### Close existing terminal and start a new one

## Install Python

[How to install Python on Ubuntu](https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/)

``` bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9-full
```

## GPU installation

### Verify you have a CUDA capable GPU
``` bash
lspci | grep -i nvidia
```

[Check NVIDIA Cuda installation](https://www.cyberciti.biz/faq/how-to-find-the-nvidia-cuda-version/)

[Medium - Install Nvidia (Driver, CUDA, cuDNN)](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1)

[NVIDIA Cuda installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

[Cuda 11.2 download archive](https://developer.nvidia.com/cuda-11.2.2-download-archive)

## Install cuDNN

[NVIDIA cudNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download)

[cuDNN download archive](https://developer.nvidia.com/rdp/cudnn-archive)

### Set the following environment variables for Ubuntu 18.04, Cuda 11.2 and cuDNN 8.1.1.x

``` bash
export cudnn_version=8.1.1.*
export cuda_version=cuda11.2
export OS=ubuntu1804
```

### For Ubuntu follow steps described in 2.3.4.1 from the NVIDIA cudNN Documentation above

``` bash
wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 

sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
```

## Updata .bashrc

### Add the following commands in ~/.bashrc

``` bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export CUDA_HOME=/usr/local/cuda
```
