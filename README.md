# AWS EMR Cluster Setup for Wine Prediction Project

This README provides step-by-step instructions for setting up an EMR cluster, configuring access, and preparing data storage for the **wine quality prediction** project.

---

## Prerequisites
- AWS account
- PuTTY (for SSH connections)
- PuTTYgen (to convert `.pem` to `.ppk`)
- Datasets (`ValidationDataset.csv` and `TrainingDataset.csv`)

---

## Step 1: Create an EMR Cluster

1. Log in to the AWS Management Console.
2. Navigate to the **EMR** service and select **EMR on EC2 clusters**.
3. On the Clusters page, click **Create Cluster**.
4. Provide a name for the cluster (e.g., `wine_prediction_1112`).
5. Select **Spark Interactive** under the Application Bundle options, which includes:
   - Spark 3.5.2
   - Livy 0.8.0
   - Jupyter Enterprise Gateway 2.6.0
6. Ensure the Amazon EMR release version is set to `emr-7.5.0`.
7. Optional: Enable additional components like Hadoop and Hive if needed.
8. Review the configuration summary and click **Create Cluster**.

---

## Step 2: Create a Key Pair

1. Navigate to the **Key Pairs** section in the EC2 service.
2. Create a new key pair with a custom name (e.g., `rr_key_pair`) and select the `.pem` file format.
3. Download the key file to your local system for later use.

---

## Step 3: Convert `.pem` to `.ppk` for PuTTY

1. Open **PuTTYgen**.
2. Load the `.pem` file and save it as a `.ppk` file.
3. Use the `.ppk` file to configure PuTTY for SSH connections.

---

## Step 4: Configure IAM Roles

1. Navigate to the **IAM Roles** section in the AWS Management Console.
2. Assign the necessary roles to the EMR cluster as required.

---

## Step 5: Modify Security Group Rules

1. Go to the **EC2 Instances** page.
2. Locate the instance associated with the EMR cluster (e.g., Master node).
3. Click on the **ElasticMapReduce-Master** security group.
4. In the **Inbound Rules** section, click **Edit inbound rules**.
5. Add the following rules:
   - Port **22** for SSH access
   - Port **4040** for Spark Web UI access
6. Save the rules.

---

## Step 6: Create an S3 Bucket

1. Navigate to the **S3** service in AWS.
2. Click **Create Bucket**.
3. Name the bucket (e.g., `rr-programming-assignment-2`) and click **Create bucket**.
4. Upload datasets:
   - Click **Add Files**.
   - Select `ValidationDataset.csv` and `TrainingDataset.csv` from your local system.
   - Click **Upload** to store the datasets in the bucket.

---

## Step 7: Connect to the Master Node via PuTTY

1. Open **PuTTY**.
2. Go to the **SSH** section under **Auth**.
3. Select **Credentials** and provide the path to the `.ppk` file.
4. Click **Open** to establish the SSH connection.

---

## Step 8: Configure AWS Credentials on the Master Node

1. Once connected, run the following commands in the terminal:
   ```bash
   # mkdir .aws
   # touch .aws/credentials
   # vi .aws/credentials
