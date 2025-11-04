<h1 align="center">Mind Map Studio Backend</h1>
<p align="center"><p align="center">
        <br>
        <a href="https://dialx.ai/">
          <img src="https://dialx.ai/dialx_logo.svg" alt="About DIALX">
        </a>
    </p>
<h4 align="center">
    <a href="https://discord.gg/ukzj9U9tEe">
        <img src="https://img.shields.io/static/v1?label=AI%20DIALX%20Community%20in&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</h4>

- [Overview](#overview)
- [Developer environment](#developer-environment)
    - [IDE configuration](#IDE-configuration)
- [Run locally](#Run-locally)
    - [Make on Windows](#Make-on-Windows)
- [Environment Variables](#environment-variables)

---

## Overview

The project is a backend part of the DIAL Mind Map Studio. 

Mind Map enables users to access information through an interactive knowledge graph and natural language. The application pulls data from various sources, including documents, URLs, and other data inputs and then presents it on UI as a interactive knowledge graph, facilitating intuitive and engaging user interactions with information.

<p align="center">
  <a href="https://www.youtube.com/watch?v=XYZfWeGdFcE">
    <img src="https://img.youtube.com/vi/XYZfWeGdFcE/0.jpg" alt="Check the demo">
  </a>
  <br>
  <em>Click to watch the demo video</em>
</p>

**[Read more about DIAL Mind Map Studio](https://docs.dialx.ai/tutorials/user-guide#mind-maps)**

---

## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=2.1.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running.

### IDE configuration

> The recommended IDE is [VSCode](https://code.visualstudio.com/).
> Open the project in VSCode and install the recommended extensions.
>
> The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

> Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).
>
> Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
> install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

---

## Run locally

Run the development server locally:

```sh
make serve
```

### Make on Windows

As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

```sh
winget install GnuWin32.Make
```

For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

---

## Environment Variables

The **Mind Map Studio** application uses environment variables to configure authentication, API connections, and theming settings. Below is a list of environment variables used in this project.

| Variable                            |        Default         | Description                                                                         |
|-------------------------------------|:----------------------:|-------------------------------------------------------------------------------------|
| `DIAL_URL`                          |    Required to set     | URL of the core DIAL server                                                         |
| `RAG_MODEL`                         |   gpt-4o-2024-05-13    | The deployment model name for the rag part.                                         |
| `GENERATOR_MODEL`                   |   gpt-4.1-2025-04-14   | The deployment model name for the graph generator part                              |
| `DESCRIPTION_INDEX_DEPLOYMENT_NAME` | gpt-4o-mini-2024-07-18 | The deployment model name to generate descriptions for the images from the sources. |
