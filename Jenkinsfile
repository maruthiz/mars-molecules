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

docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
docker rmi ${DOCKER_IMAGE} || true

docker build --no-cache -t ${DOCKER_IMAGE}:latest .

docker run -d --name ${CONTAINER_NAME} -p ${APP_PORT}:${APP_PORT} ${DOCKER_IMAGE}:latest

docker ps | grep ${CONTAINER_NAME} || echo "WARNING: Container not running!"

docker logs ${CONTAINER_NAME}
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

if ! docker images | grep ${DOCKER_IMAGE}; then
  echo "ERROR: Image ${DOCKER_IMAGE} not found. Cannot push to Docker Hub."
  exit 1
fi

echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin

docker tag ${DOCKER_IMAGE}:latest ${DOCKER_HUB_REPO}:latest
docker push ${DOCKER_HUB_REPO}:latest

docker logout
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
            bat """
                powershell -Command "if (Test-Path 'temp_ssh_key.pem') { try { Remove-Item -Path 'temp_ssh_key.pem' -Force -ErrorAction SilentlyContinue } catch { Write-Host 'Could not remove key file, continuing anyway' } }" || true
            """
            cleanWs()
        }
    }
}


