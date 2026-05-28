# Base RAPIDS
FROM rapidsai/base:24.06-cuda12.2-py3.10
SHELL ["/bin/bash", "-lc"]

# Sempre como root para evitar PermissionError no /opt/conda
USER root

# (Opcional) acelerar o conda quando precisar resolver algo
RUN conda install -n base -y -c conda-forge conda-libmamba-solver && \
    conda config --system --set solver libmamba && \
    conda config --system --set channel_priority strict

# Locks (seus arquivos)
COPY conda-spec-linux-64.txt /tmp/conda-spec.txt
COPY requirements-pip.txt     /tmp/requirements-pip.txt

# Cria o ambiente exatamente do spec e instala pip mínimo
RUN conda create -y -n salt --file /tmp/conda-spec.txt && \
    conda run -n salt pip install --no-cache-dir -r /tmp/requirements-pip.txt && \
    conda clean -afy

# Usa o env como padrão no PATH
ENV CONDA_DEFAULT_ENV=salt
ENV PATH=/opt/conda/envs/salt/bin:$PATH

# Workspace e permissões “amigáveis” (root continua sendo o usuário)
WORKDIR /workspace
RUN mkdir -p /workspace && chmod -R a+rwX /workspace
