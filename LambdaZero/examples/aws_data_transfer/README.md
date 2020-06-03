# Download data

You can access data for example via wget:

```
wget -O /filepath\_local/filename "url"
```

See /urls subfolder for the list of available files (urls). 


# Upload data

To do so you need to have the following: 

* key.pem file
* username (e.g. ubuntu)
* ip (e.g. 0.0.0.0) 

Upload process comprises following steps:

1. Transfer file from your machine to EC2 instance.
2. Connect to EC2 instance.
3. Transfer file from EC2 instance to S3 bucket (lambdazero).
4. Make file on S3 bucket publicly available (for download).

## Transfer file from your machine to EC2 instance

Launch from console on your machine:

```
scp -i key.pem /filepath\_local/filename username@ip:/filepath\_remote
``` 

## Connect to EC2 instance

```
ssh -i key.pem username@ip
```

## Transfer file from EC2 instance to S3 bucket (lambdazero)

Launch from EC2 instance: 

```
aws s3 cp /filepath\_on\_ec2/filename s3://lambdazero/filepath\_on\_s3/filename
```

## Make file on S3 bucket publicly available (for download)

Launch from EC2 instance:

```
aws s3 presign s3://lambdazero/filepath\_on\_s3/filename
```

