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
        
        // Optimization flags
        DOCKER_BUILDKIT = '1' // Enable BuildKit for more efficient builds
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
                        
                        // Enhanced cleanup script to free more disk space
                        writeFile file: 'cleanup_commands.sh', text: """#!/bin/bash
echo 'Cleaning up disk space...'
echo 'Current disk usage before cleanup:'
df -h

# Check Docker daemon status and restart if needed
echo 'Ensuring Docker daemon is running...'
if ! systemctl is-active --quiet docker; then
    sudo systemctl restart docker
    sleep 5
fi

# Stop and remove all running containers
echo 'Removing ALL existing containers...'
docker stop \$(docker ps -a -q) 2>/dev/null || true
docker rm \$(docker ps -a -q) 2>/dev/null || true

# Remove ALL images
echo 'Removing ALL existing images...'
docker rmi \$(docker images -q) -f 2>/dev/null || true

# Prune everything in Docker with extreme aggression
echo 'Aggressive Docker pruning...'
docker system prune -af --volumes

# Remove Docker's own log files
echo 'Cleaning Docker log files...'
sudo sh -c 'truncate -s 0 /var/lib/docker/containers/*/*-json.log' 2>/dev/null || true

# Clean up logs more aggressively
echo 'Cleaning up logs...'
sudo find /var/log -type f -name "*.log" -exec truncate -s 0 {} \\; 2>/dev/null || true
sudo find /var/log -type f -name "*.gz" -delete 2>/dev/null || true
sudo find /var/log -type f -name "*.1" -delete 2>/dev/null || true
sudo journalctl --vacuum-time=1d 2>/dev/null || true

# Clean package manager caches
echo 'Cleaning package manager caches...'
sudo apt-get clean
sudo apt-get autoremove -y

# Clean up /tmp and other directories
echo 'Cleaning temporary directories...'
sudo rm -rf /tmp/* /var/tmp/* 2>/dev/null || true

# Clean home directory caches
echo 'Cleaning home directory caches...'
rm -rf ~/.cache/* 2>/dev/null || true

# If git repository exists, clean it
if [ -d "\$HOME/molecules" ]; then
  echo "Cleaning git repository..."
  cd "\$HOME/molecules"
  git clean -fdx
  git gc --aggressive --prune=all
fi

# Clear swap and cache to free memory
echo 'Flushing swap and cache...'
sudo swapoff -a && sudo swapon -a
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo 'Current disk usage after cleanup:'
df -h
echo 'Available memory after cleanup:'
free -h
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
                        
                        // Check disk space before deploying
                        bat """
                            ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "df -h / | grep -v Filesystem | awk '{print \\$5}' | tr -d '%'" > disk_usage.txt
                        """
                        
                        // Read disk usage percentage
                        def diskUsage = bat(script: "type disk_usage.txt", returnStdout: true).trim()
                        echo "Current disk usage: ${diskUsage}%"
                        
                        // If disk usage is already too high (>80%), run another cleanup or fail
                        if (diskUsage.toInteger() > 80) {
                            echo "Disk usage is still high at ${diskUsage}%. Running aggressive cleanup..."
                            bat """
                                ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "~/cleanup_commands.sh" || exit 1
                            """
                        }
                        
                        // Create optimized Dockerfile for deployment
                        writeFile file: 'optimized_dockerfile.txt', text: """
# Base image for building dependencies
FROM debian:bullseye-slim AS build

# Install only necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    python3-dev \\
    python3-pip \\
    libboost-all-dev \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Final lightweight image
FROM python:3.9-slim

# Copy only necessary files from build stage
COPY --from=build /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 3000

# Start application
CMD ["python3", "app.py"]
"""
                        
                        // Create enhanced .dockerignore
                        writeFile file: 'dockerignore.txt', text: """
# Git
.git
.gitignore
.github

# Python
**/__pycache__
**/*.py[cod]
**/*$py.class
**/*.so
**/.Python
**/env/
**/build/
**/develop-eggs/
**/dist/
**/downloads/
**/eggs/
**/lib/
**/lib64/
**/parts/
**/sdist/
**/var/
**/wheels/
**/*.egg-info/
**/.installed.cfg
**/*.egg

# Unit test / coverage reports
**/htmlcov/
**/.tox/
**/.coverage
**/.coverage.*
**/cache
**/nosetests.xml
**/coverage.xml
**/*.cover
**/.hypothesis/

# Documentation
**/docs/
**/*.md
**/LICENSE
**/README*

# Editor directories and files
**/.idea
**/.vscode
**/*.swp
**/*.swo
**/.DS_Store
"""
                        
                        // Write the deployment script with optimizations
                        writeFile file: 'deploy_commands.sh', text: """#!/bin/bash
set -e  # Exit on any error

# Proper path handling for the molecules directory
REPO_DIR="\$HOME/molecules"

# Monitor disk space throughout the process
check_disk_space() {
    echo "Current disk usage:"
    df -h /
    
    # Get disk usage percentage
    USAGE=\$(df -h / | grep -v Filesystem | awk '{print \$5}' | tr -d '%')
    
    if [ \$USAGE -gt 90 ]; then
        echo "CRITICAL: Disk usage is at \${USAGE}%!"
        echo "Running emergency cleanup..."
        docker system prune -af --volumes
        sudo apt-get clean
        sudo journalctl --vacuum-time=1d
    fi
}

check_disk_space

# Check if repository directory exists, if so update it, otherwise clone
if [ -d "\$REPO_DIR" ]; then
  echo "Repository exists, pulling latest changes..."
  cd "\$REPO_DIR"
  
  # Clean repository to minimize space usage
  git clean -fdx  # Remove all untracked files
  git reset --hard HEAD  # Reset any local changes
  git fetch origin
  git checkout -f main
  git reset --hard origin/main
else
  echo "Cloning repository..."
  # Shallow clone with depth=1 to save disk space
  git clone --depth=1 https://github.com/maruthiz/mars-molecules.git "\$REPO_DIR"
fi

cd "\$REPO_DIR"
echo 'Building Docker image from Dockerfile...'

# Log the directory contents for debugging
echo "Contents of directory:"
ls -la

# Create or update .dockerignore to exclude unnecessary files
echo "Creating comprehensive .dockerignore to reduce context size..."
cat > .dockerignore << 'EOF'
$(cat /home/${EC2_USER}/dockerignore.txt)
EOF

# Check if we should use an optimized Dockerfile
if [ ! -f "Dockerfile.optimized" ]; then
  echo "Creating optimized Dockerfile..."
  cp -f /home/${EC2_USER}/optimized_dockerfile.txt Dockerfile.optimized
  
  # Only replace the original if it's safe (has backup)
  cp -f Dockerfile Dockerfile.original || true
  cp -f Dockerfile.optimized Dockerfile
fi

# Verify Dockerfile exists
if [ ! -f "Dockerfile" ]; then
  echo "ERROR: Dockerfile not found!"
  exit 1
fi

check_disk_space

# Build with optimizations to reduce space usage
echo "Building Docker image with space optimizations..."
docker build \\
  --no-cache \\
  --force-rm \\
  --compress \\
  --memory-swap -1 \\
  --build-arg BUILDKIT_INLINE_CACHE=1 \\
  -t ${DOCKER_IMAGE} . || { 
    echo "Docker build failed with standard options, attempting with minimal options..."
    # If first build fails, try with minimal options
    docker build --no-cache -t ${DOCKER_IMAGE} . || { 
      echo "Docker build failed completely"
      exit 1
    }
  }

check_disk_space

echo 'Removing any existing containers with the same name...'
# Check if container exists before trying to stop/remove
if [ \$(docker ps -a -q -f name=${CONTAINER_NAME} | wc -l) -gt 0 ]; then
  echo "Stopping existing container..."
  docker stop ${CONTAINER_NAME} || true
  echo "Removing existing container..."
  docker rm ${CONTAINER_NAME} || true
fi

check_disk_space

echo 'Starting new container...'
docker run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE} || { 
  echo "Failed to start container"
  docker logs ${CONTAINER_NAME}
  exit 1
}
echo 'Container started successfully!'

# Clean up build artifacts to save space
echo "Cleaning up build artifacts..."
docker system prune -f
docker image prune -f

# Verify container is running
if [ \$(docker ps -q -f name=${CONTAINER_NAME} | wc -l) -eq 0 ]; then
    echo "Container failed to start properly"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

# Display container information
echo "Container information:"
docker ps -a | grep ${CONTAINER_NAME}

# Show disk usage after deployment
check_disk_space
"""
                        
                        // Transfer files with binary mode to preserve line endings
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" deploy_commands.sh ${SSH_USER}@%EC2_HOST%:~/deploy_commands.sh || exit 1
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" optimized_dockerfile.txt ${SSH_USER}@%EC2_HOST%:~/optimized_dockerfile.txt || exit 1
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" dockerignore.txt ${SSH_USER}@%EC2_HOST%:~/dockerignore.txt || exit 1
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

# Final cleanup after successful push
echo "Final cleanup after successful push..."
docker system prune -af --volumes
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
            
            // Send success notification
            script {
                def recipients = emailextrecipients([[$class: 'DevelopersRecipientProvider']])
                if (recipients) {
                    emailext (
                        subject: "SUCCESS: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                        body: """<p>SUCCESS: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
                        <p>Check console output at &QUOT;<a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
                        recipientProviders: [[$class: 'DevelopersRecipientProvider']]
                    )
                }
            }
        }
        failure {
            echo "Deployment failed! Check the logs for details."
            
            // Run emergency cleanup on failure
            script {
                withCredentials([
                    string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                    sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                ]) {
                    try {
                        bat """
                            ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${SSH_USER}@%EC2_HOST% "~/cleanup_commands.sh" || true
                        """
                    } catch (Exception e) {
                        echo "Emergency cleanup failed: ${e.message}"
                    }
                }
                
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

