---
name: manage-server
description: >-
  Orchestrates starting, stopping, and checking the status of the entire
  Gravity stack (Docker PostgreSQL/PostGIS DB, FastAPI backend server, and
  Vite frontend dev server).
---

# Manage Server Skill

## Overview
This local skill provides commands to easily manage the startup, shutdown, and status verification of the Gravity (SpotSync AI) services on Windows.

It automates:
1. Starting/stopping the PostGIS PostgreSQL database container via `docker-compose`.
2. Starting/stopping the FastAPI Backend API server (`ai_search_v10.py` on Port 8000).
3. Starting/stopping the Vite Frontend dev server (Port 5173).

## Dependencies
- **Docker Desktop**: Must be installed and running on Windows.
- **Python Virtual Environment**: `.venv` folder containing the packages listed in `requirements.txt`.
- **Node.js**: Installed to execute `npm run dev` in the `frontend` directory.

## Quick Start
To manage the servers, run the helper python script using the virtual environment interpreter:

```bash
# Start all services
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py start

# Check current status
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py status

# Stop all services
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py stop
```

## Utility Scripts
The skill is CLI-based and uses `scripts/manage_server.py` with three subcommands:

### 1. `start`
Starts the PostGIS Docker container, the FastAPI backend on port 8000, and the Vite frontend on port 5173. Logs for backend and frontend are piped to `logs/backend.log` and `logs/frontend.log` respectively.
```bash
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py start
```

### 2. `status`
Displays a clean summary of whether each component is currently running or stopped.
```bash
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py status
```

### 3. `stop`
Gracefully halts all components. It kills any processes occupying ports 8000 (backend) and 5173 (frontend) using Windows `taskkill` and stops the Docker container via `docker-compose down`.
```bash
.venv\Scripts\python.exe .agents\skills\manage-server\scripts\manage_server.py stop
```

## Common Mistakes
1. **Docker Desktop not running**: Running `start` when Docker Desktop is closed will show a warning. Make sure the Docker daemon is up.
2. **Ports already in use**: If other processes occupy ports 8000 or 5173, the script will skip starting them. You can use the `stop` command to clear those ports first.
