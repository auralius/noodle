// Touch screen library with X Y and Z (pressure) readings as well
// as oversampling to avoid 'bouncing'
// (c) ladyada / adafruit
// Code under MIT License

#ifndef _TOUCHSCREEN_KBV_H_
#define _TOUCHSCREEN_KBV_H_
#include <stdint.h>

class TSPoint_kbv {
    public:
        TSPoint_kbv(void);
        TSPoint_kbv(int16_t x, int16_t y, int16_t z);

        bool operator==(TSPoint_kbv);
        bool operator!=(TSPoint_kbv);

        int16_t x, y, z;
};

class TouchScreen_kbv {
    public:
        TouchScreen_kbv(uint8_t xp, uint8_t yp, uint8_t xm, uint8_t ym);
        TouchScreen_kbv(uint8_t xp, uint8_t yp, uint8_t xm, uint8_t ym, uint16_t rx);

//        bool isTouching(void);
        uint16_t pressure(void);
        int readTouchY();
        int readTouchX();
        TSPoint_kbv getPoint();
        int16_t pressureThreshhold;

    private:
        uint8_t _yp, _ym, _xm, _xp;
        uint16_t _rxplate;
};

#endif

