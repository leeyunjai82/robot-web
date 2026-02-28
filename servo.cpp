#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>

#define BAUDRATE_VAL  B115200
#define MOTOR_RANGE_VAL {350, 150}
#define NUM_MOTORS    2
int fd;
int motor_pos[NUM_MOTORS]    = {0, };
int motor_offset[NUM_MOTORS] = {0, };
int motor_range[NUM_MOTORS]  = MOTOR_RANGE_VAL;

// ===================== Serial =====================
int set_interface_attribs(int fd, int speed, int parity)
{
  struct termios tty;
  memset(&tty, 0, sizeof tty);
  if (tcgetattr(fd, &tty) != 0) { printf("error %d from tcgetattr", errno); return -1; }
  cfsetospeed(&tty, speed);
  cfsetispeed(&tty, speed);
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_iflag &= ~IGNBRK;
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  tty.c_cc[VMIN]  = 0;
  tty.c_cc[VTIME] = 1;
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | PARODD);
  tty.c_cflag |= parity;
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;
  if (tcsetattr(fd, TCSANOW, &tty) != 0) { printf("error %d from tcsetattr", errno); return -1; }
  return 0;
}

void set_blocking(int fd, int should_block)
{
  struct termios tty;
  memset(&tty, 0, sizeof tty);
  if (tcgetattr(fd, &tty) != 0) { printf("error %d from tggetattr", errno); return; }
  tty.c_cc[VMIN]  = should_block ? 1 : 0;
  tty.c_cc[VTIME] = 1;
  if (tcsetattr(fd, TCSANOW, &tty) != 0) printf("error %d setting term attributes", errno);
}

int devOpen()
{
  struct termios options;
  fd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY);
  if (fd == -1) { printf("not opened ttyACM0\n"); return -1; }
  tcgetattr(fd, &options);
  options.c_iflag &= ~(INLCR | IGNCR | ICRNL | IXON | IXOFF);
  options.c_oflag &= ~(ONLCR | OCRNL);
  options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tcsetattr(fd, TCSANOW, &options);
  set_interface_attribs(fd, BAUDRATE_VAL, 0);
  set_blocking(fd, 0);
  return 0;
}

void devClose() { close(fd); }

// ===================== Motor Commands =====================
int SetTarget(unsigned char channel, int value)
{
  if (channel >= NUM_MOTORS) { printf("Error > Channel should be 0-%d, was: %d\n", NUM_MOTORS-1, channel); return 0; }

  motor_pos[channel] = value;
  unsigned short target = (unsigned short)((motor_pos[channel] + motor_offset[channel] + 1500) * 4);
  unsigned short max    = (1500 + motor_range[channel]) * 4;
  unsigned short min    = (1500 - motor_range[channel]) * 4;

  if (target > max) { target = max; printf("Warn > Target clamped [%d]: %d\n",  channel,  motor_range[channel]); }
  if (target < min) { target = min; printf("Warn > Target clamped [%d]: %d\n",  channel, -motor_range[channel]); }

  unsigned char command[] = {0x84, channel, (unsigned char)(target & 0x7F), (unsigned char)(target >> 7 & 0x7F)};
  if (write(fd, command, sizeof(command)) == -1) { printf("error writing\n"); return -1; }
  return 0;
}

int SetMultiTarget(int *pTarget)
{
  for (int ch = 0; ch < NUM_MOTORS; ch++)
    motor_pos[ch] = pTarget[ch];

  unsigned char command[3 + NUM_MOTORS * 2];
  command[0] = 0x9F;
  command[1] = NUM_MOTORS;
  command[2] = 0;
  int idx = 3;

  for (int ch = 0; ch < NUM_MOTORS; ch++)
  {
    unsigned short target = (unsigned short)((motor_pos[ch] + motor_offset[ch] + 1500) * 4);
    unsigned short max    = (1500 + motor_range[ch]) * 4;
    unsigned short min    = (1500 - motor_range[ch]) * 4;
    if (target > max) { target = max; printf("Warn > Target clamped [%d]: %d\n",  ch,  motor_range[ch]); }
    if (target < min) { target = min; printf("Warn > Target clamped [%d]: %d\n",  ch, -motor_range[ch]); }
    command[idx++] = target & 0x7F;
    command[idx++] = target >> 7 & 0x7F;
  }

  if (write(fd, command, sizeof(command)) == -1) { printf("error writing\n"); return -1; }
  return 0;
}

// ===================== Main =====================
int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Usage:\n");
    printf("\t%-65s\n\n", "servo mwrite pos0<-900~900> pos1<-900~900>");
    printf("\t%-65s\n\n", "servo write channel<0~1> value<-900~900>");
    return 0;
  }

  devOpen();

  // servo mwrite [pos0] [pos1]
  if (strcmp(argv[1], "mwrite") == 0)
  {
    int pPos[NUM_MOTORS];
    for (int i = 0; i < NUM_MOTORS; i++) pPos[i] = atoi(argv[i + 2]);
    SetMultiTarget(pPos);
  }
  // servo write [channel] [pos]
  else if (strcmp(argv[1], "write") == 0)
  {
    SetTarget(atoi(argv[2]), atoi(argv[3]));
  }
  else
  {
    printf(" ! No such command: %s\n", argv[1]);
    printf("Usage:\n");
    printf("\t%-65s\n\n", "servo mwrite pos0<-900~900> pos1<-900~900>");
    printf("\t%-65s\n\n", "servo write channel<0~1> value<-900~900>");
  }

  devClose();
  return 0;
}
