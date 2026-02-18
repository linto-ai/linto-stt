def notifyLintoDeploy(service_name, tag, commit_sha) {
    echo "Deployed ${service_name}:${tag} (commit: ${commit_sha})"
}

def buildDockerImage(backend, image_name, version, changedFiles, commit_sha) {
    boolean has_changed = changedFiles.contains("linto_stt/backends/${backend}/")

    // Shared paths that affect all backends
    def sharedPaths = [
        'Dockerfile',
        'docker-entrypoint.sh',
        'pyproject.toml',
        'uv.lock',
        'linto_stt/__init__.py',
        'linto_stt/http_server/',
        'linto_stt/websocket/',
        'linto_stt/punctuation/',
    ]
    for (path in sharedPaths) {
        if (changedFiles.contains(path)) {
            has_changed = true
            break
        }
    }

    if (has_changed) {
        echo "Building Docker image for ${image_name} with version ${version} (backend: ${backend})"

        script {
            def image = docker.build(image_name, "--build-arg SERVICE_NAME=${backend} -f Dockerfile .")

            docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                image.push(version)
                if (!version.contains('latest-unstable')) {
                    image.push('latest')
                }
            }
        }

        if (env.BRANCH_NAME == 'master') {
            notifyLintoDeploy(backend, version, commit_sha)
        }
    }
}

pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO_KALDI   = "lintoai/linto-stt-kaldi"
        DOCKER_HUB_REPO_WHISPER = "lintoai/linto-stt-whisper"
        DOCKER_HUB_REPO_NEMO    = "lintoai/linto-stt-nemo"
        DOCKER_HUB_REPO_KYUTAI  = "lintoai/linto-stt-kyutai"
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
                    def commit_sha = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
                    echo "Changed files: ${changedFiles}"

                    def version = sh(
                        returnStdout: true,
                        script: "grep '^version' pyproject.toml | head -1 | sed 's/.*\"\\(.*\\)\"/\\1/'"
                    ).trim()

                    buildDockerImage('nemo',    env.DOCKER_HUB_REPO_NEMO,    version, changedFiles, commit_sha)
                    buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles, commit_sha)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI,   version, changedFiles, commit_sha)
                    // buildDockerImage('kyutai',  env.DOCKER_HUB_REPO_KYUTAI,  version, changedFiles, commit_sha)
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
                    def commit_sha = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
                    echo "Changed files: ${changedFiles}"

                    def version = 'latest-unstable'

                    buildDockerImage('nemo',    env.DOCKER_HUB_REPO_NEMO,    version, changedFiles, commit_sha)
                    buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles, commit_sha)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI,   version, changedFiles, commit_sha)
                    // buildDockerImage('kyutai',  env.DOCKER_HUB_REPO_KYUTAI,  version, changedFiles, commit_sha)
                }
            }
        }
    }
}
