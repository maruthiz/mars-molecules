pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'molecules-app'
        DOCKER_TAG = 'latest'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    bat "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    // Start the container
                    bat "docker run -d --name test-container -p 3000:3000 ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    
                    // Wait for application to start
                    bat "timeout /t 30"
                    
                    // Basic health check
                    bat "curl http://localhost:3000/"
                }
            }
            post {
                always {
                    // Cleanup test container
                    bat "docker stop test-container || true"
                    bat "docker rm test-container || true"
                }
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                script {
                    // Add your deployment steps here
                    // For example, pushing to Docker registry or deploying to server
                    echo "Deploying to production..."
                }
            }
        }
    }

    post {
        always {
            // Cleanup
            bat "docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true"
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}