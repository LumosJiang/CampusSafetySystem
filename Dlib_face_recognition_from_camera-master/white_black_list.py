import os
from datetime import time
from flask import Flask, jsonify, request
from flask_cors import CORS  # 导入 CORS 扩展
from face_finally_life import  face_recognizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})


@app.route('/get_whitelist_users', methods=['GET'])
def get_whitelist_users():
    whitelist_dir = 'data/data_faces_from_camera'
    users = []

    for folder_name in os.listdir(whitelist_dir):
        if os.path.isdir(os.path.join(whitelist_dir, folder_name)):
            user_name = folder_name.split('_')[-1]
            if not user_name.endswith('f'):  # 排除以 'f' 结尾的 user_name
                users.append(user_name)

    return jsonify(users)

@app.route('/get_blacklist_users', methods=['GET'])
def get_blacklist_users():
    whitelist_dir = 'data/data_faces_blacklist'
    users = []

    for folder_name in os.listdir(whitelist_dir):
        if os.path.isdir(os.path.join(whitelist_dir, folder_name)):
            user_name = folder_name.split('_')[-1]
            users.append(user_name)

    return jsonify(users)


@app.route('/add_to_blacklist', methods=['POST'])
def add_to_blacklist():
    try:
        data = request.json
        user_name = data.get('user_name')

        if user_name:
            blacklist_dir = 'data/data_faces_blacklist'
            user_folder = os.path.join(blacklist_dir, user_name)
            os.makedirs(user_folder, exist_ok=True)

            # Rename user folders in data_faces_from_camera
            faces_dir = 'data/data_faces_from_camera'
            for folder_name in os.listdir(faces_dir):
                if user_name in folder_name:
                    old_path = os.path.join(faces_dir, folder_name)
                    new_folder_name = folder_name + 'f'
                    new_path = os.path.join(faces_dir, new_folder_name)
                    os.rename(old_path, new_path)
            app.logger.info(f'User {user_name} added to blacklist')

            return jsonify({'status': 'success', 'message': f'User {user_name} added to blacklist'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'User name is required'}), 400
    except Exception as e:
        app.logger.error(f'Error adding user to blacklist: {str(e)}')
        return jsonify({'status': 'error', 'message': 'Failed to add user to blacklist'}), 500

# 解除黑名单的路由
@app.route('/remove_from_blacklist', methods=['POST'])
def remove_from_blacklist():
    data = request.json
    user_name = data.get('user_name')
    print(user_name)
    if user_name:
        blacklist_dir = 'data/data_faces_blacklist'
        user_folder = os.path.join(blacklist_dir, user_name)

        if os.path.exists(user_folder):
            os.rmdir(user_folder)  # 删除用户文件夹

            # 检查并重命名 data_faces_from_camera 中的文件夹
            faces_dir = 'data/data_faces_from_camera'
            for folder_name in os.listdir(faces_dir):
                if folder_name.endswith(f'_{user_name}f'):
                    old_path = os.path.join(faces_dir, folder_name)
                    new_folder_name = folder_name[:-1]  # 去掉末尾的 'f'
                    new_path = os.path.join(faces_dir, new_folder_name)
                    os.rename(old_path, new_path)
                    app.logger.info(f'Renamed {folder_name} to {new_folder_name} in whitelist')
            return jsonify({'status': 'success', 'message': f'User {user_name} removed from blacklist'}), 200
        else:
            return jsonify({'status': 'error', 'message': f'User {user_name} not found in blacklist'}), 404
    else:
        return jsonify({'status': 'error', 'message': 'User name is required'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5002)
