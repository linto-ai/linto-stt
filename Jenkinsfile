def buildDockerfile(main_folder, dockerfilePath, image_name, version, changedFiles) {
    if (changedFiles.contains(main_folder) || changedFiles.contains('celery_app') || changedFiles.contains('http_server') || changedFiles.contains('websocket') || changedFiles.contains('document')) {
        echo "Building Dockerfile for ${image_name} with version ${version} (using ${dockerfilePath})"

        script {
            def image = docker.build(image_name, "-f ${dockerfilePath} .")

            docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                if (version  == 'latest-unstable') {
                    image.push('latest-unstable')
                } else {
                    image.push('latest')
                    image.push(version)
                }
            }
        }
    }
}

pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO_KALDI   = "lintoai/linto-stt-kaldi"
        DOCKER_HUB_REPO_WHISPER = "lintoai/linto-stt-whisper"
    }
    
    stages {
        stage('Docker build for master branch') {
            when {
                branch 'master'
            }
            steps {
                echo 'Publishing latest'
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    echo "My changed files: ${changedFiles}"
                    
                    version_kaldi = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' kaldi/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    version_whisper = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' whisper/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    buildDockerfile('kaldi', 'kaldi/Dockerfile', env.DOCKER_HUB_REPO_KALDI, version_kaldi, changedFiles)
                    buildDockerfile('whisper', 'whisper/Dockerfile.ctranslate2', env.DOCKER_HUB_REPO_WHISPER, version_whisper, changedFiles)
                }
            }
        }

        stage('Docker build for next (unstable) branch') {
            when {
                branch 'next'
            }
            steps {
                echo 'Publishing unstable'
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    echo "My changed files: ${changedFiles}"
                    
                    version = 'latest-unstable'

                    buildDockerfile('kaldi', 'kaldi/Dockerfile', env.DOCKER_HUB_REPO_KALDI, version, changedFiles)
                    buildDockerfile('whisper', 'whisper/Dockerfile.ctranslate2', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles)
                }
            }
        }
    }
}