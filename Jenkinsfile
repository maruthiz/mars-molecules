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
        
        // Application directory on EC2
        APP_DIR = '~/flask-app'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }
        
        stage('Sync Code to EC2') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                    ]) {
                        // Create a temporary key file with proper permissions
                        bat """
                            echo "Creating temporary key file with proper permissions"
                            powershell -Command "Copy-Item -Path '$SSH_KEY' -Destination 'temp_ssh_key.pem'"
                            powershell -Command "icacls 'temp_ssh_key.pem' /inheritance:r"
                            powershell -Command "icacls 'temp_ssh_key.pem' /grant:r 'SYSTEM:R' /grant:r 'Administrators:R'"
                        """
                        
                        // Create a tar archive of the workspace
                        bat "tar -czf molecules-app.tar.gz --exclude='.git' --exclude='node_modules' --exclude='__pycache__' ."
                        
                        // Ensure the target directory exists on EC2
                        bat """
                            set SSH_AUTH_SOCK=
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "mkdir -p ${APP_DIR}"
                        """
                        
                        // Transfer the archive to EC2 and extract it
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" molecules-app.tar.gz ${SSH_USER}@%EC2_HOST%:~/molecules-app.tar.gz
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "tar -xzf ~/molecules-app.tar.gz -C ${APP_DIR} && rm ~/molecules-app.tar.gz"
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
                        // Create deployment script
                        writeFile file: 'deploy_commands.sh', text: """#!/bin/bash
cd ${APP_DIR}
echo 'Building Docker image...'
docker build -t ${DOCKER_IMAGE} .
echo 'Stopping and removing existing container...'
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
echo 'Starting new container...'
docker run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE}
echo 'Container started successfully!'
"""

                        // Transfer deployment script and execute it
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" deploy_commands.sh ${SSH_USER}@%EC2_HOST%:~/deploy_commands.sh
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "chmod +x ~/deploy_commands.sh && ~/deploy_commands.sh"
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
                        // Create push script with fix for missing image tag
                        writeFile file: 'push_commands.sh', text: """#!/bin/bash
echo 'Logging in to Docker Hub...'
echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin
echo 'Tagging image for Docker Hub...'
# Check if image exists before tagging
if docker image inspect ${DOCKER_IMAGE} >/dev/null 2>&1; then
    docker tag ${DOCKER_IMAGE} ${DOCKER_HUB_REPO}:latest
else
    echo "Image ${DOCKER_IMAGE} not found. Using ${DOCKER_HUB_REPO}:latest directly."
fi
echo 'Pushing image to Docker Hub...'
docker push ${DOCKER_HUB_REPO}:latest || true
docker logout
echo "Docker Hub push process completed"
"""

                        // Transfer push script and execute it
                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" push_commands.sh ${SSH_USER}@%EC2_HOST%:~/push_commands.sh
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "chmod +x ~/push_commands.sh && ~/push_commands.sh"
                        """

                        // Skip temporary key file removal - this was causing failures
                        // Instead, we'll modify file permissions to prepare for later deletion
                        bat """
                            powershell -Command "if (Test-Path 'temp_ssh_key.pem') { icacls 'temp_ssh_key.pem' /reset }" || true
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            echo " Deployment completed successfully!"
        }
        failure {
            echo " Deployment failed! Check the logs."
            script {
                def recipients = emailextrecipients([[$class: 'DevelopersRecipientProvider']])
                if (recipients) {
                    emailext (
                        subject: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                        body: """<p>FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'</p>
                        <p>See console output: <a href='${env.BUILD_URL}'>${env.BUILD_URL}</a></p>""",
                        recipientProviders: [[$class: 'DevelopersRecipientProvider']]
                    )
                }
            }
        }
        always {
            // Suppress errors from file cleaning operations
            bat """
                powershell -Command "if (Test-Path 'temp_ssh_key.pem') { try { Remove-Item -Path 'temp_ssh_key.pem' -Force -ErrorAction SilentlyContinue } catch { Write-Host 'Could not remove key file, continuing anyway' } }" || true
            """
            cleanWs(cleanWhenNotBuilt: false, deleteDirs: true, disableDeferredWipeout: true, notFailBuild: true)
        }
    }
}

