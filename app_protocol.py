import os
import shutil
import json
from pathlib import Path
from threading import Lock
from cachetools import LRUCache
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from pybot_module import pyBot
from uuid import uuid4
from datetime import timedelta

# Base directory for instance configs
BASE_SAVE_DIR = Path(os.getenv("BOT_SAVE_DIR", "D:/.yourchatbot")).resolve()
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LRU cache to limit loaded bot instances
bot_cache = LRUCache(maxsize=10)
bot_lock = Lock()
conv_histories = LRUCache(maxsize=1000)

def load_config(instance_name: str) -> dict:
    """Load instance configuration from disk."""
    cfg_path = BASE_SAVE_DIR / instance_name / 'instance_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config for '{instance_name}' not found.")
    return json.loads(cfg_path.read_text())


def get_bot(instance_name: str) -> pyBot:
    """Get or load a pyBot for the given instance name, using LRU eviction."""
    with bot_lock:
        bot = bot_cache.get(instance_name)
        if bot is None:
            cfg = load_config(instance_name)
            bot = pyBot(
                instance_name=cfg['instance_name'],
                llm_name=cfg['llm_name'],
                save_dir=cfg.get('save_dir', str(BASE_SAVE_DIR)),
                force_rebuild=cfg.get('force_rebuild', False),
                update=cfg.get('update', False),
                target_url=cfg['target_url']
            )
            bot_cache[instance_name] = bot
        return bot

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-me')
CORS(app,
     supports_credentials=True,
     resources={r"/*": {"origins": ["http://localhost:5173"]}})
app.permanent_session_lifetime = timedelta(hours=1)
@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/chat/<name>/reset', methods=['POST'])
def reset_session(name):
    sid = session.get('sid')
    if sid and (name, sid) in conv_histories:
        conv_histories.pop((name, sid), None)
    return jsonify({"message": "Chat history cleared."}), 200

# List current instances
@app.route('/instances', methods=['GET'])
def list_instances():
    instances = []
    for inst_dir in BASE_SAVE_DIR.iterdir():
        cfg_file = inst_dir / 'instance_config.json'
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                instances.append({'name': cfg.get('instance_name'), 'config': cfg})
            except Exception:
                continue
    return jsonify(instances), 200

@app.route('/instances', methods=['POST'])
def create_instance():
    data = request.get_json(force=True)
    client_name = data.get('client_name')
    name = data['instance_name']
    llm = data.get('llm', 'phi')
    target_url = data.get('url') or data.get('target_url')
    force = data.get('force_rebuild', False)
    update = data.get('update', False)

    # Prevent duplicates
    with bot_lock:
        if name in bot_cache or (BASE_SAVE_DIR / name).exists():
            return jsonify({'error': 'Instance already exists'}), 400

    # Build pyBot args
    bot_args = {
        'instance_name': name,
        'llm_name':      llm,
        'save_dir':      str(BASE_SAVE_DIR),
        'force_rebuild': force,
        'update':        update,
        'target_url':    target_url
    }

    # Instantiate and configure
    bot = pyBot(**bot_args)
    bot.set_llm(
        friendliness=data.get('friendliness'),
        ooc=data.get('oocMessage') or data.get('ooc'),
        contact=data.get('contactInfo') or data.get('contactValues')
    )

    # Cache
    with bot_lock:
        bot_cache[name] = bot

    # Persist config including client_name
    config = {'client_name': client_name, **bot_args}
    cfg_path = BASE_SAVE_DIR / name / 'instance_config.json'
    cfg_path.write_text(json.dumps(config, indent=2))

    # Also save initial LLM settings to meta.json
    meta = {}
    # gather initial settings from frontend
    if data.get('friendliness') is not None:
        meta['friendliness'] = data['friendliness']
    if data.get('oocMessage') or data.get('ooc'):
        meta['ooc'] = data.get('oocMessage') or data.get('ooc')
    if data.get('contactInfo') or data.get('contactValues'):
        meta['contact'] = data.get('contactInfo') or data.get('contactValues')
    # write meta.json alongside
    meta_path = BASE_SAVE_DIR / name / 'meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))

    # Dynamic URL
    base = request.url_root.rstrip('/')
    chat_url = f"{base}/chat/{name}"
    return jsonify({'message': f"Instance '{name}' created", 'url': chat_url}), 201

@app.route('/chat/<name>', methods=['POST'])
def chat(name):
    # 1) Identify or generate a session ID for this visitor
    sid = session.get('sid')
    if not sid:
        sid = str(uuid4())
        session['sid'] = sid

    key = (name, sid)

    # 2) Get the last few turns (or start fresh)
    hist = conv_histories.get(key, [])
    print(hist)

    # 3) Pull the incoming query
    data = request.get_json(force=True)
    query = data.get('query')
    if not query:
        return jsonify({'error': "Missing 'query' field"}), 400

    # 4) Load or 404 the bot instance
    try:
        bot = get_bot(name)
    except FileNotFoundError:
        return jsonify({'error': 'Instance not found'}), 404

    # 5) Call converse, passing in the prior history
    try:
        answer = bot.converse(query, history=hist)
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

    # 6) Record just this session’s last 5 turns
    hist.append((query, answer))
    conv_histories[key] = hist[-5:]

    # 7) Return the AI’s response
    return jsonify({'response': answer}), 200

@app.route('/instances/<name>', methods=['DELETE'])
def destroy_instance(name):
    # Evict from cache and destroy to release file locks
    with bot_lock:
        bot = bot_cache.pop(name, None)
    if bot:
        try:
            bot.destroy()
        except Exception:
            pass
    # Remove on disk
    config_dir = BASE_SAVE_DIR / name
    if config_dir.exists():
        shutil.rmtree(config_dir)
        return jsonify({'message': f"Instance '{name}' destroyed."}), 200
    return jsonify({'error': 'Instance not found'}), 404

@app.route('/instances/<name>/settings', methods=['PATCH'])
def update_settings(name):
    try:
        bot = get_bot(name)
        in_cache = True
    except FileNotFoundError:
        in_cache = False
    data = request.get_json(force=True)

    # Load or initialize meta settings
    inst_dir = BASE_SAVE_DIR / name
    meta_path = inst_dir / 'meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    # Only update provided non-empty fields
    kwargs = {}
    if 'friendliness' in data and data['friendliness'] is not None:
        kwargs['friendliness'] = data['friendliness']
        meta['friendliness'] = data['friendliness']
    if 'ooc' in data and data['ooc']:
        kwargs['ooc'] = data['ooc']
        meta['ooc'] = data['ooc']
    if 'contact' in data and data['contact']:
        kwargs['contact'] = data['contact']
        meta['contact'] = data['contact']

    # Persist LLM settings to meta.json
    meta_path.write_text(json.dumps(meta, indent=2))

    # Apply to bot if loaded
    if in_cache and kwargs:
        bot.set_llm(**kwargs)

    return jsonify({'message': 'Settings updated.'}), 200

@app.route('/instances/<name>/rebuild', methods=['POST'])
def rebuild(name):
    """
    Rebuilds an existing chatbot instance: destroys current artifacts, rebuilds from saved config.
    """
    # 1️⃣ Load existing config
    try:
        raw_cfg = load_config(name)
    except FileNotFoundError:
        return jsonify({'error': 'Instance not found'}), 404

    # 2️⃣ Destroy current bot and cache entry
    with bot_lock:
        old_bot = bot_cache.pop(name, None)
    if old_bot:
        try:
            old_bot.destroy()
        except Exception:
            pass

    # 3️⃣ Prepare rebuild arguments
    rebuild_args = {
        'instance_name': raw_cfg['instance_name'],
        'llm_name':      raw_cfg['llm_name'],
        'save_dir':      raw_cfg.get('save_dir', str(BASE_SAVE_DIR)),
        'force_rebuild': True,
        'update':        True,
        'target_url':    raw_cfg['target_url']
    }

    # 4️⃣ Instantiate new bot (this will crawl & build anew)
    new_bot = pyBot(**rebuild_args)
    # Reapply LLM settings
    new_bot.set_llm(
        friendliness=raw_cfg.get('friendliness'),
        ooc=raw_cfg.get('ooc'),
        contact=raw_cfg.get('contact')
    )

    # 5️⃣ Cache and persist rebuild flags back to config file
    with bot_lock:
        bot_cache[name] = new_bot
    # Save original client_name and other frontend fields
    save_cfg = {
        'client_name':  raw_cfg.get('client_name'),
        'instance_name': raw_cfg['instance_name'],
        'llm_name':      raw_cfg['llm_name'],
        'save_dir':      raw_cfg.get('save_dir', str(BASE_SAVE_DIR)),
        'force_rebuild': False,  # reset flags after rebuild
        'update':        False,
        'target_url':    raw_cfg['target_url']
    }
    cfg_path = BASE_SAVE_DIR / name / 'instance_config.json'
    cfg_path.write_text(json.dumps(save_cfg, indent=2))

    return jsonify({'message': f"Instance '{name}' rebuilt successfully."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
