pipeline {
    agent any

    environment {
        EC2_USER = 'ubuntu'
        EC2_HOST = credentials('ec2-host-ip')
        DOCKER_IMAGE = 'molecular-property-app'
        CONTAINER_NAME = 'molecular_container'
        APP_PORT = '3000'
        DOCKER_HUB_REPO = "marslec/${DOCKER_IMAGE}"
        APP_DIR = '~/molecules'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
                bat 'git log -1 --pretty=format:"%h - %an, %ar : %s"'
            }
        }

        stage('Sync Code to EC2') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                    ]) {
                        bat """
                            powershell -Command "Copy-Item -Path '$SSH_KEY' -Destination 'temp_ssh_key.pem'"
                            powershell -Command "icacls 'temp_ssh_key.pem' /inheritance:r"
                            powershell -Command "icacls 'temp_ssh_key.pem' /grant:r 'SYSTEM:R' /grant:r 'Administrators:R'"
                        """

                        bat """
                            set SSH_AUTH_SOCK=
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "rm -rf ${APP_DIR}/* && mkdir -p ${APP_DIR}"
                        """

                        bat "tar -czf molecules-app.tar.gz --exclude='.git' --exclude='__pycache__' ."

                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" molecules-app.tar.gz ${SSH_USER}@%EC2_HOST%:~/molecules-app.tar.gz
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "tar -xzf ~/molecules-app.tar.gz -C ${APP_DIR} && rm ~/molecules-app.tar.gz"
                        """

                        bat """
                            set SSH_AUTH_SOCK=
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "ls -la ${APP_DIR} && if [ -f ${APP_DIR}/Dockerfile ]; then echo 'Dockerfile found'; else echo 'Dockerfile NOT found!'; fi"
                        """
                    }
                }
            }
        }

        stage('Setup Docker Permissions') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ec2-host-ip', variable: 'EC2_HOST'),
                        sshUserPrivateKey(credentialsId: 'ec2-docker-deploy', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
                    ]) {
                        writeFile file: 'setup_docker_permissions.sh', text: """#!/bin/bash
set -e
# Check if user is in docker group
if ! groups | grep -q docker; then
  echo "Adding user to docker group..."
  sudo usermod -aG docker \$USER
  echo "User added to docker group. Note: You may need to log out and back in for changes to take effect."
  # Try to apply group membership without logging out
  exec sg docker -c "id"
fi
"""

                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" setup_docker_permissions.sh ${SSH_USER}@%EC2_HOST%:~/setup_docker_permissions.sh
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "chmod +x ~/setup_docker_permissions.sh && ~/setup_docker_permissions.sh || echo 'Permission setup may require a server restart'"
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
                        writeFile file: 'deploy_commands.sh', text: """#!/bin/bash
set -ex
cd ${APP_DIR}

echo 'Listing contents of molecules directory:'
ls -la

if [ ! -f Dockerfile ]; then
  echo "ERROR: Dockerfile not found!"
  exit 1
fi

# Try with sudo if direct docker command fails
docker_cmd() {
  if ! docker \$@; then
    echo "Trying with sudo..."
    sudo docker \$@
  fi
}

docker_cmd stop ${CONTAINER_NAME} || true
docker_cmd rm ${CONTAINER_NAME} || true
docker_cmd rmi ${DOCKER_IMAGE} || true

docker_cmd build --no-cache -t ${DOCKER_IMAGE}:latest .

docker_cmd run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE}:latest

docker_cmd ps | grep ${CONTAINER_NAME} || echo "WARNING: Container not running!"

docker_cmd logs ${CONTAINER_NAME}
"""

                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" deploy_commands.sh ${SSH_USER}@%EC2_HOST%:~/deploy_commands.sh
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "chmod +x ~/deploy_commands.sh && ~/deploy_commands.sh"
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
                        writeFile file: 'push_commands.sh', text: """#!/bin/bash
set -ex

# Try with sudo if direct docker command fails
docker_cmd() {
  if ! docker \$@; then
    echo "Trying with sudo..."
    sudo docker \$@
  fi
}

if ! docker_cmd images | grep ${DOCKER_IMAGE}; then
  echo "ERROR: Image ${DOCKER_IMAGE} not found. Cannot push to Docker Hub."
  exit 1
fi

echo '${DOCKER_HUB_PASSWORD}' | docker_cmd login -u '${DOCKER_HUB_USERNAME}' --password-stdin

docker_cmd tag ${DOCKER_IMAGE}:latest ${DOCKER_HUB_REPO}:latest
docker_cmd push ${DOCKER_HUB_REPO}:latest

docker_cmd logout
"""

                        bat """
                            set SSH_AUTH_SOCK=
                            scp -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" push_commands.sh ${SSH_USER}@%EC2_HOST%:~/push_commands.sh
                            ssh -o StrictHostKeyChecking=no -i "temp_ssh_key.pem" ${SSH_USER}@%EC2_HOST% "chmod +x ~/push_commands.sh && ~/push_commands.sh"
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
            echo "Deployment failed! Check the logs."
        }
        always {
            // Fix the error in the PowerShell command with proper quoting and error handling
            bat '''
                powershell -Command "if (Test-Path 'temp_ssh_key.pem') { Remove-Item -Path 'temp_ssh_key.pem' -Force -ErrorAction SilentlyContinue }"
            '''
            cleanWs()
        }
    }
}


