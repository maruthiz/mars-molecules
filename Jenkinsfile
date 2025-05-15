pipeline {
    agent any
    
    options {
        timeout(time: 1, unit: 'HOURS')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

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
                sshagent(['ec2-docker-deploy']) {
                    // Create a tar archive of the workspace
                    bat "tar -czf molecules-app.tar.gz --exclude='.git' --exclude='node_modules' --exclude='__pycache__' ."
                    
                    // Ensure the target directory exists on EC2
                    bat """
                        ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no ${EC2_USER}@%EC2_HOST% "mkdir -p ${APP_DIR}"
                    """
                    
                    // Transfer the archive to EC2 and extract it
                    bat """
                        scp -o StrictHostKeyChecking=no molecules-app.tar.gz ${EC2_USER}@%EC2_HOST%:~/molecules-app.tar.gz
                        ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no ${EC2_USER}@%EC2_HOST% "tar -xzf ~/molecules-app.tar.gz -C ${APP_DIR} && rm ~/molecules-app.tar.gz"
                    """
                }
            }
        }

        stage('Deploy to EC2') {
            steps {
                sshagent(['ec2-docker-deploy']) {
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
                        scp -o StrictHostKeyChecking=no deploy_commands.sh ${EC2_USER}@%EC2_HOST%:~/deploy_commands.sh
                        ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no ${EC2_USER}@%EC2_HOST% "chmod +x ~/deploy_commands.sh && ~/deploy_commands.sh"
                    """
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                sshagent(['ec2-docker-deploy']) {
                    withCredentials([usernamePassword(credentialsId: 'docker_hub', usernameVariable: 'DOCKER_HUB_USERNAME', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                        // Create push script
                        writeFile file: 'push_commands.sh', text: """#!/bin/bash
echo 'Logging in to Docker Hub...'
echo '${DOCKER_HUB_PASSWORD}' | docker login -u '${DOCKER_HUB_USERNAME}' --password-stdin
echo 'Tagging image for Docker Hub...'
docker tag ${DOCKER_IMAGE} ${DOCKER_HUB_REPO}:latest
echo 'Pushing image to Docker Hub...'
docker push ${DOCKER_HUB_REPO}:latest
docker logout
"""

                        // Transfer push script and execute it
                        bat """
                            scp -o StrictHostKeyChecking=no push_commands.sh ${EC2_USER}@%EC2_HOST%:~/push_commands.sh
                            ssh -o ConnectTimeout=30 -o ConnectionAttempts=3 -o StrictHostKeyChecking=no ${EC2_USER}@%EC2_HOST% "chmod +x ~/push_commands.sh && ~/push_commands.sh"
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
            // Clean up temporary files before cleaning workspace
            bat """
                powershell -Command "try { if (Test-Path 'deploy_commands.sh') { Remove-Item -Path 'deploy_commands.sh' -Force -ErrorAction SilentlyContinue } } catch {}"
                powershell -Command "try { if (Test-Path 'push_commands.sh') { Remove-Item -Path 'push_commands.sh' -Force -ErrorAction SilentlyContinue } } catch {}"
                powershell -Command "try { if (Test-Path 'molecules-app.tar.gz') { Remove-Item -Path 'molecules-app.tar.gz' -Force -ErrorAction SilentlyContinue } } catch {}"
            """
            cleanWs(disableDeferredWipeout: true, deleteDirs: true)
        }
    }
}
