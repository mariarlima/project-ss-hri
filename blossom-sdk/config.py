#!/usr/bin/env python
# -*- coding: utf-8 -*-

ROBOT_330_LAB = {
    "controllers": {
        "port": "/dev/tty.usbserial-FT5NUSRG",
        "protocol": 2,
        "baudrate": 1000000,
        "drivemode": 12,         # time-based profile and torque on by goal update 
        "blocking": True,
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 1200,       # code for XL-330
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        # "ears": {
        #     "id": 5,
        #     "type": 1200,
        #     "angle_limit": [-150.0, 150.0]
        # }
    }
}

ROBOT_330 = {
    "controllers": {
        "port": "/dev/tty.usbserial-FT9MIR1N",
        "protocol": 2,
        "baudrate": 1000000,
        "drivemode": 0,
        "blocking": True
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 1200,       # code for XL-330
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "ears": {
            "id": 5,
            "type": 1200,
            "angle_limit": [50, 130.0],
        }
    }
}

ROBOT_330_TIME = {
    "controllers": {
        "port": "/dev/tty.usbserial-FT9MIR1N",
        "protocol": 2,
        "baudrate": 1000000,
        "drivemode": 12,         # time-based profile and torque on by goal update 
        "blocking": True,
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 1200,       # code for XL-330
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "ears": {
            "id": 5,
            "type": 1200,
            "angle_limit": [50, 130.0],
        }
    }
}


ROBOT_320 = {
    "controllers": {
        "port": "/dev/tty.usbserial-FTAAMOPF",      # port for XL-320 on Emily's computer 
        "protocol": 2,
        "baudrate": 1000000,
        "drivemode": 12,
        "blocking": True
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 350,       # code for XL-320
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        # "ears": {
        #     "id": 5,
        #     "type": 350,
        #     "angle_limit": [50, 130.0],
        # }
    }
}

ROBOT_330_RPI = {
    "controllers": {
        "port": "/dev/ttyUSB0",
        "protocol": 2,
        "baudrate": 1000000,
        "drivemode": 0,
        "blocking": True
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 1200,       # code for XL-330
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 1200,
            "angle_limit": [-150.0, 150.0],
        },
        "ears": {
            "id": 5,
            "type": 1200,
            "angle_limit": [50, 130.0],
        }
    }
}


ROBOT_320_RPI = {
    "controllers": {
        "port": "/dev/ttyUSB0",      # port for XL-320 on Emily's computer 
        "protocol": 2,
        "baudrate": 1000000,
        "blocking": True
    },
    "motors": {
        "base": {
            "id": 4,
            "type": 350,       # code for XL-320
            "angle_limit": [-150.0, 150.0],
        },
        "tower_1": {
            "id": 1,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_2": {
            "id": 2,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        "tower_3": {
            "id": 3,
            "type": 350,
            "angle_limit": [-150.0, 150.0],
        },
        "ears": {
            "id": 5,
            "type": 350,
            "angle_limit": [50, 130.0],
        }
    }
}
