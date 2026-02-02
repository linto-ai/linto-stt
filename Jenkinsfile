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

def buildDockerfile(main_folder, dockerfilePath, image_name, version, changedFiles, commit_sha, buildArgs = "") {
    boolean has_changed = changedFiles.contains(main_folder) || changedFiles.contains('celery_app') || changedFiles.contains('http_server') || changedFiles.contains('websocket') || changedFiles.contains('document')

    if (main_folder == "kaldi" || main_folder == "nemo") {
        // Kaldi also depends on recasepunc
        has_changed = has_changed || changedFiles.contains('punctuation')
    }

    def contextLocation = "."
    if (dockerfilePath == "kyutai/Dockerfile") {
        contextLocation = "./kyutai"
    }

    if (has_changed) {
        echo "Building Dockerfile for ${image_name} with version ${version} (using ${dockerfilePath})"

        script {
            def image = docker.build(image_name, "-f ${dockerfilePath} ${buildArgs} ${contextLocation}")

            docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                image.push(version)
                if (!version.contains('latest-unstable')) {
                    image.push('latest')
                }
            }

            // Notify linto-deploy after successful push (only for master branch)
            if (!version.contains('latest-unstable')) {
                // Extract service name from full image name (remove "lintoai/" prefix)
                def service_name = image_name.replace('lintoai/', '')
                notifyLintoDeploy(service_name, version, commit_sha)
            }
        }
    }
}

pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO_KALDI   = "lintoai/linto-stt-kaldi"
        DOCKER_HUB_REPO_WHISPER = "lintoai/linto-stt-whisper"
        DOCKER_HUB_REPO_NEMO = "lintoai/linto-stt-nemo"
        DOCKER_HUB_REPO_KYUTAI_WRAPPER = "lintoai/linto-stt-kyutai-wrapper"
        DOCKER_HUB_REPO_KYUTAI_MOSHI_CUDA = "lintoai/kyutai-moshi-stt-server-cuda"
        DOCKER_HUB_REPO_KYUTAI_MOSHI_CPU = "lintoai/kyutai-moshi-stt-server-cpu"
    }

    stages {
        stage('Docker build for master branch') {
            when {
                branch 'master'
            }
            steps {
                echo 'Publishing latest'
                dir('kyutai') {
                    sh 'git submodule update --init --recursive'
                }
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    def commit_sha = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
                    echo "My changed files: ${changedFiles}"

                    version_kaldi = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' kaldi/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    version_whisper = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' whisper/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    version_nemo = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' nemo/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    version_kyutai = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' kyutai/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    buildDockerfile('kaldi', 'kaldi/Dockerfile', env.DOCKER_HUB_REPO_KALDI, version_kaldi, changedFiles, commit_sha)
                    buildDockerfile('whisper', 'whisper/Dockerfile.ctranslate2', env.DOCKER_HUB_REPO_WHISPER, version_whisper, changedFiles, commit_sha)
                    buildDockerfile('nemo', 'nemo/Dockerfile', env.DOCKER_HUB_REPO_NEMO, version_nemo, changedFiles, commit_sha)
                    buildDockerfile('kyutai', 'kyutai/Dockerfile.wrapper', env.DOCKER_HUB_REPO_KYUTAI_WRAPPER, version_kyutai, changedFiles, commit_sha)
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CUDA, version_kyutai, changedFiles, commit_sha, '--target runtime')
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CPU, version_kyutai, changedFiles, commit_sha, '--target runtime-cpu')
                }
            }
        }

        stage('Docker build for next (unstable) branch') {
            when {
                branch 'next'
            }
            steps {
                echo 'Publishing unstable'
                dir('kyutai') {
                    sh 'git submodule update --init --recursive'
                }
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    echo "My changed files: ${changedFiles}"

                    version = 'latest-unstable'

                    buildDockerfile('kaldi', 'kaldi/Dockerfile', env.DOCKER_HUB_REPO_KALDI, version, changedFiles, '')
                    buildDockerfile('whisper', 'whisper/Dockerfile.ctranslate2', env.DOCKER_HUB_REPO_WHISPER, version, changedFiles, '')
                    buildDockerfile('nemo', 'nemo/Dockerfile', env.DOCKER_HUB_REPO_NEMO, version, changedFiles, '')

                    buildDockerfile('kyutai', 'kyutai/Dockerfile.wrapper', env.DOCKER_HUB_REPO_KYUTAI_WRAPPER, version, changedFiles, '')
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CUDA, "${version}-ampere", changedFiles, '', '--no-cache --target runtime --build-arg CUDARC_COMPUTE=86')
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CUDA, "${version}-ada", changedFiles, '', '--no-cache --target runtime --build-arg CUDARC_COMPUTE=89')
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CUDA, "${version}-hopper", changedFiles, '', '--no-cache --target runtime --build-arg CUDARC_COMPUTE=90')
                    buildDockerfile('kyutai', 'kyutai/Dockerfile', env.DOCKER_HUB_REPO_KYUTAI_MOSHI_CPU, version, changedFiles, '', '--target runtime-cpu')
                }
            }
        }
    }
}
