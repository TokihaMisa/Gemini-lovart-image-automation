# AI 代理与开发维护指南 (Agent Instructions)

这份文档是为开发者以及后续接手的 AI 助手（如 Gemini、Cursor、Copilot 等）准备的备忘录。

## ⚠️ 发版与 OTA 自动更新必读 (Critical Update Protocol)

当你或者 AI 代理修改了项目的核心代码，并准备发布新版本供运营人员使用时，**必须严格遵守以下自动更新发版流程**，否则客户端的“检查更新”功能将无法生效：

1. **升级本地版本号**：
   - 打开 `version.py`。
   - 修改 `VERSION = "1.x.x"`（必须递增，如从 `1.0.0` 改为 `1.0.1`）。

2. **升级云端配置文件**：
   - 打开根目录下的 `version.json`。
   - 将 `"version": "1.x.x"` 修改为与 `version.py` 相同的新版本号。
   - 填写 `"changelog": "本次更新的具体内容"`。
   - 确保 `"url"` 指向即将上传的最新 `update.zip` 的下载直链。

3. **执行编译打包**：
   - 在项目根目录双击运行 `build_exe.bat`（或在终端运行它）。
   - 等待 PyInstaller 打包完成，前往 `dist/Lovart自动化助手` 目录。

4. **上传更新包**：
   - 将打包好的 `dist/Lovart自动化助手` 文件夹压缩成 `update.zip`。
   - 将这个 `update.zip` 上传到您配置的服务器或 GitHub Releases 中。

5. **推送更新触发指令**：
   - 使用 Git 将 `version.json` 以及修改过的代码推送到 GitHub 的 `main` 分支。
   - 推送成功后，远端的 `version.json` 就更新了。所有旧版客户端点击“检查更新”时就会全自动下载新包并完成覆盖替换。

---

> **AI 代理指令 (For AI Agents)**:
> 以后如果在对话中涉及到“新增功能”、“修复bug”并且用户暗示需要更新客户端时，请务必主动提醒用户：**“是否需要我帮您同步修改 version.py 和 version.json 的版本号以便触发自动更新？”**
