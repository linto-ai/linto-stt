def buildWhisper(image_name, version) {
    echo "Building Dockerfile for ${image_name}... with version ${version}"

    script {
        def image = docker.build(image_name, "-f whisper/Dockerfile.ctranslate2 .")

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

def buildKaldi(image_name, version) {
    echo "Building Dockerfile for ${image_name}... with version ${version}"

    script {
        def image = docker.build(image_name, "-f kaldi/Dockerfile  .")

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

pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO_KALDI   = "lintoai/linto-stt-kaldi"
        DOCKER_HUB_REPO_WHISPER = "lintoai/linto-stt-whisper"

        VERSION_KALDI = ''
        VERSION_WHISPER = ''
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
                    
                    VERSION_KALDI = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' kaldi/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    VERSION_WHISPER = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' whisper/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()
                    
                    if (changedFiles.contains('celery_app') || changedFiles.contains('http_server') || changedFiles.contains('websocket') || changedFiles.contains('document')) {
                        echo "Build kaldi version ${VERSION_KALDI}"
                        buildKaldi(env.DOCKER_HUB_REPO_KALDI, VERSION_KALDI)

                        echo "Build whisper version ${VERSION_WHISPER}"
                        buildWhisper(env.DOCKER_HUB_REPO_WHISPER, VERSION_WHISPER)
                    }else {
                        if (changedFiles.contains('kaldi')) {
                        echo "Build kaldi version ${VERSION_KALDI}"
                            buildKaldi(env.DOCKER_HUB_REPO_KALDI, VERSION_KALDI)
                        }
                        if (changedFiles.contains('whisper')) {
                            echo "Build whisper version ${VERSION_WHISPER}"
                            buildWhisper(env.DOCKER_HUB_REPO_WHISPER, VERSION_WHISPER)
                        }
                    }
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
                    
                    VERSION = 'latest-unstable'
                    
                    if (changedFiles.contains('celery_app') || changedFiles.contains('http_server') || changedFiles.contains('websocket') || changedFiles.contains('document')) {
                        echo 'Files in studio-api path are modified. Running specific build steps for studio-api...'
                        echo "Build whisper and kaldi version ${VERSION}"

                        buildKaldi(env.DOCKER_HUB_REPO_KALDI, VERSION)
                        buildWhisper(env.DOCKER_HUB_REPO_WHISPER, VERSION)
                    }else {
                        if (changedFiles.contains('kaldi')) {
                            echo "Build kaldi version ${VERSION}"
                            buildKaldi(env.DOCKER_HUB_REPO_KALDI, VERSION)
                        }
                        if (changedFiles.contains('whisper')) {
                            echo "Build whisper version ${VERSION}"
                            buildWhisper(env.DOCKER_HUB_REPO_WHISPER, VERSION)
                        }
                    }
                }
            }
        }
    }
}
