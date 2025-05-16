pipeline {
    agent any
    
    environment {
        // EC2 connection details
        EC2_USER = 'ubuntu'
        EC2_HOST = credentials('ec2-host-ip')
        
        // Docker configuration
        DOCKER_IMAGE = 'molecular-property-app'
        CONTAINER_NAME = 'nice_williams'
        APP_PORT = '3000'
        
        // Docker Hub details
        DOCKER_HUB_REPO = "marslec/${DOCKER_IMAGE}"
        
        // Build information
        BUILD_NUMBER = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Cleanup EC2') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                    ]) {
                        // Fix SSH key permissions with more restrictive settings
                        bat """
                            echo Fixing SSH key permissions with secure settings...
                            powershell -Command "icacls '$SSH_KEY' /inheritance:r"
                            powershell -Command "icacls '$SSH_KEY' /grant:r 'SYSTEM:F' /grant:r 'Administrators:F'"
                            powershell -Command "icacls '$SSH_KEY' /remove 'mars\\\\marut'"
                        """
                        
                        // Write the cleanup script to free disk space
                        writeFile file: 'cleanup_commands.sh', text: """#!/bin/bash
echo 'Cleaning up disk space...'
echo 'Current disk usage before cleanup:'
df -h

# Remove any existing molecular-property-app containers
echo 'Removing existing containers...'
if [ \$(docker ps -a -q -f name=${CONTAINER_NAME} | wc -l) -gt 0 ]; then
  docker stop ${CONTAINER_NAME} || true
  docker rm ${CONTAINER_NAME} || true
  echo 'Existing containers removed.'
fi

# Remove any existing molecular-property-app images
echo 'Removing existing images...'
if [ \$(docker images -q ${DOCKER_IMAGE} | wc -l) -gt 0 ]; then
  docker rmi ${DOCKER_IMAGE} || true
  echo 'Existing images removed.'
fi

# Remove dangling images and volumes
echo 'Removing dangling images and volumes...'
docker image prune -f
docker volume prune -f

# Remove unused Docker resources
echo 'Removing unused Docker resources...'
docker system prune -af --volumes

# Clean up apt cache
echo 'Cleaning apt cache...'
sudo apt-get clean

# Remove old logs
echo 'Removing old logs...'
sudo find /var/log -type f -name "*.gz" -delete

echo 'Current disk usage after cleanup:'
df -h
"""
                        
                        // Transfer and run the cleanup script
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" cleanup_commands.sh ${SSH_USER}@%EC2_HOST%:~/cleanup_commands.sh || exit 1
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "chmod +x ~/cleanup_commands.sh && ~/cleanup_commands.sh" || exit 1
                        """
                    }
                }
            }
        }
        
        stage('Deploy to EC2') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                    ]) {
                        // Fix SSH key permissions with more restrictive settings
                        bat """
                            echo Fixing SSH key permissions with secure settings...
                            powershell -Command "icacls '$SSH_KEY' /inheritance:r"
                            powershell -Command "icacls '$SSH_KEY' /grant:r 'SYSTEM:F' /grant:r 'Administrators:F'"
                            powershell -Command "icacls '$SSH_KEY' /remove 'mars\\\\marut'"
                        """
                        
                        // Write the deployment script to a file with error handling
                        writeFile file: 'deploy_commands.sh', text: """#!/bin/bash
set -e  # Exit on any error

# Check if repository directory exists, if not clone it
if [ ! -d "~/molecules" ]; then
  echo "Cloning repository..."
  git clone https://github.com/maruthiz/mars-molecules.git ~/molecules
else
  echo "Repository exists, pulling latest changes..."
  cd ~/molecules
  git pull
fi

cd ~/molecules
echo 'Building Docker image from Dockerfile...'
# Log the directory contents for debugging
echo "Contents of directory:"
ls -la

# Verify Dockerfile exists
if [ ! -f "Dockerfile" ]; then
  echo "ERROR: Dockerfile not found!"
  exit 1
fi

# Build with no-cache to ensure fresh build
docker build --no-cache -t ${DOCKER_IMAGE} . || { echo "Docker build failed"; exit 1; }

echo 'Removing any existing containers with the same name...'
# Check if container exists before trying to stop/remove
if [ \$(docker ps -a -q -f name=${CONTAINER_NAME} | wc -l) -gt 0 ]; then
  echo "Stopping existing container..."
  docker stop ${CONTAINER_NAME} || true
  echo "Removing existing container..."
  docker rm ${CONTAINER_NAME} || true
fi

echo 'Starting new container...'
docker run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE} || { echo "Failed to start container"; exit 1; }
echo 'Container started successfully!'

# Verify container is running
if [ \$(docker ps -q -f name=${CONTAINER_NAME} | wc -l) -eq 0 ]; then
    echo "Container failed to start properly"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

# Display container information
echo "Container information:"
docker ps -a | grep ${CONTAINER_NAME}
"""
                        
                        // Transfer the script with binary mode to preserve line endings
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" deploy_commands.sh ${SSH_USER}@%EC2_HOST%:~/deploy_commands.sh || exit 1
                        """
                        
                        // Make script executable and run it
                        bat """
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "chmod +x ~/deploy_commands.sh && ~/deploy_commands.sh" || exit 1
                        """
                    }
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER'),
                        usernamePassword(credentialsId: 'docker_hub', usernameVariable: 'DOCKER_HUB_USERNAME', passwordVariable: 'DOCKER_HUB_PASSWORD')
                    ]) {
                        // Fix SSH key permissions with more restrictive settings
                        bat """
                            echo Fixing SSH key permissions with secure settings...
                            powershell -Command "icacls '$SSH_KEY' /inheritance:r"
                            powershell -Command "icacls '$SSH_KEY' /grant:r 'SYSTEM:F' /grant:r 'Administrators:F'"
                            powershell -Command "icacls '$SSH_KEY' /remove 'mars\\\\marut'"
                        """
                        
                        // Write the Docker Hub push script to a file with better error handling
                        writeFile file: 'push_commands.sh', text: """#!/bin/bash
set -e  # Exit on any error

echo "Checking for Docker image existence..."
if [[ -z \$(docker images -q ${DOCKER_IMAGE}) ]]; then
    echo "ERROR: Image ${DOCKER_IMAGE} does not exist locally"
    echo "Available images:"
    docker images
    exit 1
fi

echo 'Logging in to Docker Hub...'
echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin || { echo "Docker login failed"; exit 1; }

echo 'Tagging image for Docker Hub...'
docker tag ${DOCKER_IMAGE} ${DOCKER_HUB_REPO}:latest || { echo "Docker tag failed"; exit 1; }

echo 'Pushing image to Docker Hub...'
docker push ${DOCKER_HUB_REPO}:latest || { echo "Docker push failed"; exit 1; }

# Add a tag with build number for versioning
echo 'Adding build number tag...'
docker tag ${DOCKER_IMAGE} ${DOCKER_HUB_REPO}:build-${BUILD_NUMBER:-1} || { echo "Docker tag with build number failed"; exit 1; }
docker push ${DOCKER_HUB_REPO}:build-${BUILD_NUMBER:-1} || { echo "Docker push with build number failed"; exit 1; }

echo 'Logout from Docker Hub'
docker logout

echo "Successfully pushed image to Docker Hub: ${DOCKER_HUB_REPO}:latest and ${DOCKER_HUB_REPO}:build-${BUILD_NUMBER:-1}"
"""
                        
                        // Transfer the script with binary mode to preserve line endings
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" push_commands.sh ${SSH_USER}@%EC2_HOST%:~/push_commands.sh || exit 1
                        """
                        
                        // Make script executable and run it
                        bat """
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "chmod +x ~/push_commands.sh && ~/push_commands.sh" || exit 1
                        """
                    }
                }
            }
        }
    }
    
    post {
        success {
            echo "Deployment completed successfully!"
        }
        failure {
            echo "Deployment failed! Check the logs for details."
            script {
                def recipients = emailextrecipients([[$class: 'DevelopersRecipientProvider']])
                if (recipients) {
                    emailext (
                        subject: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                        body: """<p>FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
                        <p>Check console output at &QUOT;<a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
                        recipientProviders: [[$class: 'DevelopersRecipientProvider']]
                    )
                }
            }
        }
        always {
            cleanWs()
        }
    }
}
