from datetime import datetime
import json
import os
import platform
import smtplib
import socket
from email.message import EmailMessage


def info_equip() -> dict:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "desconegut"
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "gpu": gpu,
    }


def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _carregar_env_file():
    """Carrega variables des d'un fitxer .env si existeix.

    No sobreescriu variables ja definides a l'entorn del procés.
    """
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    ]

    env_path = next((path for path in candidates if os.path.isfile(path)), None)
    if not env_path:
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def registrar_hiperparametres(model, ruta_model="models", suffix="", opponents=None, total_timesteps=None, initial_model_path=None):
    if not os.path.exists(ruta_model):
        os.makedirs(ruta_model)

    def _extract_opponent_names(opponents_list):
        if opponents_list is None:
            return []
        return [type(op).__name__ for op in opponents_list]

    opponent_names = _extract_opponent_names(opponents)

    if not opponent_names:
        try:
            env = model.get_env()
            if env is not None and getattr(env, "envs", None):
                base_env = env.envs[0]
                current_opponents = getattr(base_env.unwrapped, "agents", [])
                opponent_names = _extract_opponent_names(current_opponents)
        except Exception:
            opponent_names = []

    hiperparams = {
        "learning_rate": model.learning_rate,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "clip_range": model.clip_range(1.0) if callable(model.clip_range) else model.clip_range,
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "target_kl": model.target_kl,
        "seed": model.seed,
        "device": str(model.device),
        "policy_class": model.policy_class.__name__,
        "policy_kwargs": model.policy_kwargs,
        "opponent_agents": ", ".join(opponent_names) if opponent_names else "-",
        "total_timesteps": total_timesteps,
        "initial_model_path": initial_model_path,
        **info_equip(),
    }

    hiperparams_path = os.path.join(ruta_model, "hiperparametres" + suffix + ".json")
    _ensure_parent_dir(hiperparams_path)
    with open(hiperparams_path, "w", encoding="utf-8") as f:
        json.dump(hiperparams, f, indent=4)
    print(f"Hiperparàmetres guardats a: {hiperparams_path}")


def preparar_execucio(run_name):
    run_dir = os.path.join("runs", run_name)
    paths = {
        "run_dir": run_dir,
        "logs_dir": os.path.join(run_dir, "logs"),
        "monitor_dir": os.path.join(run_dir, "monitor"),
        "config_dir": os.path.join(run_dir, "config"),
        "emissions_dir": os.path.join(run_dir, "emissions"),
        "model_dir": os.path.join(run_dir, "model"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def enviar_mail(subject, body, to_email=None, from_email=None, smtp_host=None, smtp_port=None, username=None, password=None, use_tls=True):
    """
    Envia un correu electrònic via SMTP.

    Configuració per defecte via variables d'entorn:
      - MAIL_TO
      - MAIL_FROM
      - MAIL_SMTP_HOST
      - MAIL_SMTP_PORT
      - MAIL_SMTP_USER
      - MAIL_SMTP_PASSWORD
      - MAIL_SMTP_TLS (true/false)
    """
    _carregar_env_file()

    to_email = to_email or os.environ.get("MAIL_TO")
    from_email = from_email or os.environ.get("MAIL_FROM")
    smtp_host = smtp_host or os.environ.get("MAIL_SMTP_HOST")
    smtp_port = smtp_port or os.environ.get("MAIL_SMTP_PORT", "587")
    username = username or os.environ.get("MAIL_SMTP_USER")
    password = password or os.environ.get("MAIL_SMTP_PASSWORD")

    env_tls = os.environ.get("MAIL_SMTP_TLS")
    if env_tls is not None:
        use_tls = env_tls.strip().lower() in {"1", "true", "yes", "on"}

    if not to_email:
        raise ValueError("Falta 'to_email' o la variable d'entorn MAIL_TO")
    if not from_email:
        raise ValueError("Falta 'from_email' o la variable d'entorn MAIL_FROM")
    if not smtp_host:
        raise ValueError("Falta 'smtp_host' o la variable d'entorn MAIL_SMTP_HOST")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email
    message.set_content(body)

    smtp_port = int(smtp_port)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(message)

    print(f"Correu enviat a: {to_email}")