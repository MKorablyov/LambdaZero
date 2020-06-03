# Download data

You can access data for example via wget:

```
wget -O /filepath_local/filename "url"
```

See /urls subfolder for the list of available files (urls). 


# Upload data

To do so you are going to need the following (ask idt-team in Slack): 

* key.pem file
* username (e.g. ubuntu)
* ip (e.g. 0.0.0.0) 

Upload process comprises following steps:

1. Transfer file from your machine to EC2 instance.
2. Connect to EC2 instance.
3. Transfer file from EC2 instance to S3 bucket (lambdazero).
4. Make file on S3 bucket publicly available for download.

## Transfer file from your machine to EC2 instance

Launch from console on your machine:

```
scp -i key.pem /filepath_local/filename username@ip:/filepath_remote
``` 

## Connect to EC2 instance

```
ssh -i key.pem username@ip
```

## Transfer file from EC2 instance to S3 bucket (lambdazero)

Launch from EC2 instance: 

```
aws s3 cp /filepath_on_ec2/filename s3://lambdazero/filepath_on_s3/filename
```

## Make file on S3 bucket publicly available for download

Launch from EC2 instance:

```
aws s3 presign s3://lambdazero/filepath_on_s3/filename
```
