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
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
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
                            powershell -Command "icacls '${SSH_KEY}' /inheritance:r"
                            powershell -Command "icacls '${SSH_KEY}' /grant:r 'SYSTEM:F' /grant:r 'Administrators:F'"
                        """
                        
                        // Write the deployment script to a file
                        writeFile file: 'deploy_commands.sh', text: """#!/bin/bash
cd ~/flask-app
echo 'Building Docker image...'
docker build -t ${DOCKER_IMAGE} .
echo 'Stopping and removing existing container...'
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
echo 'Starting new container...'
docker run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE}
echo 'Container started successfully!'
"""
                        
                        // Transfer the script and execute it
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" deploy_commands.sh ${EC2_USER}@%EC2_HOST%:~/deploy_commands.sh
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${EC2_USER}@%EC2_HOST% "chmod +x ~/deploy_commands.sh && ~/deploy_commands.sh"
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
                            powershell -Command "icacls '${SSH_KEY}' /inheritance:r"
                            powershell -Command "icacls '${SSH_KEY}' /grant:r 'SYSTEM:F' /grant:r 'Administrators:F'"
                        """
                        
                        // Write the Docker Hub push script to a file
                        writeFile file: 'push_commands.sh', text: """#!/bin/bash
echo 'Logging in to Docker Hub...'
echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin
echo 'Tagging image for Docker Hub...'
docker tag ${DOCKER_IMAGE} ${DOCKER_HUB_REPO}:latest
echo 'Pushing image to Docker Hub...'
docker push ${DOCKER_HUB_REPO}:latest
echo 'Logout from Docker Hub'
docker logout
"""
                        
                        // Transfer the script and execute it
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "${SSH_KEY}" push_commands.sh ${EC2_USER}@%EC2_HOST%:~/push_commands.sh
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "${SSH_KEY}" ${EC2_USER}@%EC2_HOST% "chmod +x ~/push_commands.sh && ~/push_commands.sh"
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
        }
        always {
            cleanWs()
        }
    }
}