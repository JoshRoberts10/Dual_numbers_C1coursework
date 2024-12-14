# docker file adapted from lecture notes


# Use an official Python image as the base image
FROM python:3.10-slim

# Install Git
RUN apt-get update && apt-get install -y \
    git\
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app


RUN mkdir -p /app/notebooks



# Copy only the wheels from your local directory into the Docker container
COPY dual_autodiff_x/wheelhouse/*.whl /app/wheelhouse/
COPY  dist/*.whl /app/dist/


COPY Notebooks /app/notebooks

COPY tests /app/tests

COPY requirements.txt /app/

# Install required dependencies for building the package
RUN pip install --upgrade pip setuptools wheel setuptools_scm build

RUN pip install jupyter

RUN pip install -r requirements.txt

# Install the linux wheels from the wheelhouse directory
RUN pip install wheelhouse/dual_autodiff_x-1.1.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl



# Install the python from the wheelhouse directory
RUN pip install dist/dual_autodiff-1.0.6.dev26+g0993659.d20241211-py3-none-any.whl

#CMD ["bash"]

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]