```zsh
conda env create -f environment.yml
```

```zsh
conda activate YTsentimentAnalyzer
```

 mlflow server -h 0.0.0.0 -p 5000 \
  --default-artifact-root s3://mlflow-bucket-52 \
  --allowed-hosts "ec2-13-51-197-16.eu-north-1.compute.amazonaws.com:5000,13.51.197.16:5000" 