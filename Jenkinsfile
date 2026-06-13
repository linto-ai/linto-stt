def notifyLintoDeploy(service_name, tag, commit_sha) {
    echo "Notifying linto-deploy for ${service_name}:${tag} (commit: ${commit_sha})..."
    withCredentials([usernamePassword(
        credentialsId: 'linto-deploy-bot',
        usernameVariable: 'GITHUB_APP',
        passwordVariable: 'GITHUB_TOKEN'
    )]) {
        writeFile file: 'payload.json', text: "{\"event_type\":\"update-service\",\"client_payload\":{\"service\":\"${service_name}\",\"tag\":\"${tag}\",\"commit_sha\":\"${commit_sha}\"}}"
        sh 'curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" -d @payload.json https://api.github.com/repos/linto-ai/linto-deploy/dispatches'
    }
}

// Best-effort deploy of a freshly built image to the staging cluster (full CI/CD).
// Needs a Jenkins SSH credential 'staging-deploy-ssh' (key for ubuntu@bm2-3s);
// if absent the build still succeeds (push-only).
def stagingDeploy(image_name, tag) {
    try {
        withCredentials([sshUserPrivateKey(credentialsId: 'staging-deploy-ssh', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')]) {
            sh "ssh -i \$SSH_KEY -o StrictHostKeyChecking=no \$SSH_USER@163.114.159.33 'staging-deploy ${image_name} ${tag}'"
        }
    } catch (err) {
        echo "Staging auto-deploy skipped for ${image_name}:${tag} (add the 'staging-deploy-ssh' credential to enable): ${err}"
    }
}

def buildDockerImage(service_type, image_name, version, changedFiles, commit_sha, extraDeps = '', gpu = false) {
    boolean has_changed = changedFiles.contains("linto_stt/engines/${service_type}/")

    // Shared paths that affect all engines
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
        echo "Building Docker image for ${image_name} with version ${version} (service_type: ${service_type})"

        script {
            def extraArg = extraDeps ? "--build-arg EXTRA_DEPS=${extraDeps}" : ""
            def gpuArg = gpu ? "--build-arg GPU=1" : ""
            def image = docker.build(image_name, "--build-arg STT_ENGINE=${service_type} ${extraArg} ${gpuArg} -f Dockerfile .")

            docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                image.push(version)
                if (!version.contains('latest-unstable')) {
                    image.push('latest')
                }
            }
        }

        if (!version.contains('latest-unstable')) {
            def service_name = image_name.replace('lintoai/', '')
            notifyLintoDeploy(service_name, version, commit_sha)
        }
    }
}

pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO_KALDI   = "lintoai/linto-stt-kaldi"
        DOCKER_HUB_REPO_WHISPER = "lintoai/linto-stt-whisper"
        DOCKER_HUB_REPO_WHISPER_CPU = "lintoai/linto-stt-whisper-cpu"
        DOCKER_HUB_REPO_NEMO    = "lintoai/linto-stt-nemo"
        DOCKER_HUB_REPO_KYUTAI  = "lintoai/linto-stt-kyutai"
        DOCKER_HUB_REPO_KALDI_RECASEPUNC = "lintoai/linto-stt-kaldi-recasepunc"
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
                    // buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER_CPU, version, changedFiles, commit_sha)
                    buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles, commit_sha, '', true)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI,   version, changedFiles, commit_sha)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI_RECASEPUNC, version, changedFiles, commit_sha, 'recasepunc')
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
                    // buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER_CPU, version, changedFiles, commit_sha)
                    buildDockerImage('whisper', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles, commit_sha, '', true)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI,   version, changedFiles, commit_sha)
                    buildDockerImage('kaldi',   env.DOCKER_HUB_REPO_KALDI_RECASEPUNC, version, changedFiles, commit_sha, 'recasepunc')
                    // buildDockerImage('kyutai',  env.DOCKER_HUB_REPO_KYUTAI,  version, changedFiles, commit_sha)
                }
            }
        }

        // Staging builds the whisper (GPU) engine — the only STT worker running on
        // the staging cluster — and points linto-stt-whisper at it.
        stage('Docker build for staging branches') {
            when {
                branch 'staging/*'
            }
            steps {
                echo 'Building staging feature-branch image (whisper GPU, private registry, never Docker Hub)'
                script {
                    def slug = env.BRANCH_NAME.replaceFirst('^staging/', '').replaceAll('[^a-zA-Z0-9]+', '-').toLowerCase()
                    def tag = "dev-${slug}"
                    def image = docker.build("registry.staging.linto.ai/lintoai/linto-stt-whisper", "--build-arg STT_ENGINE=whisper --build-arg GPU=1 -f Dockerfile .")
                    docker.withRegistry('https://registry.staging.linto.ai', 'staging-registry-credentials') {
                        image.push(tag)
                    }
                    stagingDeploy('linto-stt-whisper', tag)
                }
            }
        }
    }
}
