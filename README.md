<!-- 启动命令 -->
docker run -d --name rag-service `
  -p 8000:8000 `
  --env-file .\.env `
  -v rag_uploads:/app/uploads `
  -v rag_chroma:/app/chroma_db `
  -v rag_logs:/app/logs `
  -v rag_session_data:/app/data `
  fastapi-rag:day13

<!--
注意：只挂数据目录，绝对不要写成 -v rag_session:/app。
挂载 /app 会用宿主卷覆盖镜像里构建好的代码，容器会一直跑旧版本，
表现为"代码改了、也重新部署了，但新功能就是不生效"。
会话库已从 <项目>/session.db 挪到 <项目>/data/session.db，所以单独挂 /app/data 即可。
-->

<!-- deploy.ps1 脚本（推荐：含部署后版本一致性校验与鉴权校验） -->
.\deploy.ps1
