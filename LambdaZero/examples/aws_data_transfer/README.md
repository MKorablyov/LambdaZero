# Download data

# TODO: see standard data description on S3 
# See /urls subfolder for the list of available files (urls).

```
# todo: need list available files

# You can access data for example via wget:
wget -O /filepath_local/filename "url"
```

 



# Upload data
# To do so you are going to need the following (ask idt-team in Slack): 
* key.pem file
* username (e.g. ubuntu)
* ip (e.g. 0.0.0.0) 
```
# Transfer file from your machine to EC2 instance.
scp -i key.pem /filepath_local/filename username@ip:/filepath_remote
# Connect to EC2 instance.
ssh -i key.pem username@ip
# Transfer file from EC2 instance to S3 bucket (lambdazero).
aws s3 cp /filepath_on_ec2/filename s3://lambdazero/filepath_on_s3/filename
# Make file on S3 bucket publicly available for download.
aws s3 presign s3://lambdazero/filepath_on_s3/filename
```

