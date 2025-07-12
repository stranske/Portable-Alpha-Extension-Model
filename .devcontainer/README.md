# Codespaces Configuration

This directory contains GitHub Codespaces configuration files that ensure your development environment is set up automatically.

## Files

- **`devcontainer.json`**: Main configuration file that defines the container, extensions, and settings
- **`setup.sh`**: Post-creation script that installs dependencies and configures the environment
- **`vscode-settings.json`**: VS Code settings specific to Codespaces

## What happens when you start a Codespace

1. GitHub creates a container based on the Python 3.11 image
2. Installs VS Code extensions automatically
3. Runs the setup script to create virtual environment and install dependencies
4. Configures port forwarding for the Streamlit dashboard (port 8501)
5. Sets up proper Python paths and environment variables

## Customization

You can modify these files to add:
- Additional VS Code extensions in `devcontainer.json`
- Extra dependencies in `setup.sh`
- Custom environment variables
- Additional port forwards

## Testing the Configuration

To test changes to the devcontainer configuration:
1. Make your changes
2. Commit and push to GitHub
3. Create a new Codespace to test the setup

The setup should complete in under 2 minutes for a fresh Codespace.
