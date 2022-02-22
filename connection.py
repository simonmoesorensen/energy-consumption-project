from typing import Dict

from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder


def get_db_engine(ssh_config: Dict, db_config: Dict):
    print('Connecting to the SSH Tunnel...')

    ssh_tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(ssh_config['SSH_HOST'], 22),
        ssh_username=ssh_config["SSH_USER"],
        ssh_password=ssh_config["SSH_PASS"],
        remote_bind_address=(db_config["DB_HOST"], db_config["DB_PORT"])
    )
    ssh_tunnel.start()

    print('Connecting to the PostgreSQL Database...')
    return create_engine(
        'postgresql://{user}:{password}@{host}:{port}/{db}'.format(
            host='localhost',
            port=ssh_tunnel.local_bind_port,
            user=db_config["DB_USER"],
            password=db_config["DB_PASS"],
            db=db_config["DB_NAME"]
        )
    )
