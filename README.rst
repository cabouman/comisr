.. docs-include-ref

comiser
========

..
    Change the number of = to match the number of characters in the project name.

package for video super-resolution

..
    Include more detailed description here.

Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@github.com:cabouman/comiser

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``comiser`` using the following commands:

            .. code-block::

                conda create --name comiser python=3.10
                conda activate comiser
                pip install -r requirements.txt

            Anytime you want to use this package, this ``comiser`` environment should be activated with the following:

            .. code-block::

                conda activate comiser


        2. *Install comiser package:*

            Navigate to the main directory ``comiser/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .


Running Demo(s)
---------------

Run any of the available demo scripts with something like the following:

    .. code-block::

        python demo/<demo_file>.py

