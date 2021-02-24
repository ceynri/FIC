import { Controller, Get, Post, UseInterceptors, UploadedFile } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { AppService } from './app.service';

@Controller('api')
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Get('uploads')
  uploadsTest() {
    return 'uploads api';
  }

  @Post('uploads')
  @UseInterceptors(FileInterceptor('file'))
  uploadFile(@UploadedFile() file) {
    return this.appService.uploadFile(file);
  }

  @Post('demo_process')
  @UseInterceptors(FileInterceptor('file'))
  demoProcess(@UploadedFile() file) {
    return this.appService.demoProcess(file);
  }
}
