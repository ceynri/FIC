import { Injectable } from '@nestjs/common';

import child_process from 'child_process';
import * as fs from 'fs';

import * as moment from 'moment';

@Injectable()
export class AppService {
  getHello(): string {
    return 'Hello World!';
  }

  uploadFile(file) {
    console.log(file);
    const date = moment().format('YYYY-MM-DD');
    // TODO: 合理的存放位置
    const storageAddress = `C:/Users/Haze/Desktop/毕设/temp/${date}/${file.originalname}`;
    const writeStream = fs.createWriteStream(storageAddress);
    writeStream.write(file.buffer);
  }

  // demoProcess(imageDto): Promise<any>  {
  //   return new Promise((resolve, reject) => {
  //     const workerProcess = child_process.exec(
  //       `python demo.py ${imageDto}`,
  //       (error, stdout, stderr) => {
  //         if (error) {
  //           console.log(error.stack);
  //           console.log(`Error code ${error.code}`);
  //           console.log(`Signal received ${error.signal}`);
  //           reject(error);
  //         }
  //         console.log(`stdout: ${stdout}`);
  //         console.log(`stderr: ${stderr}`);
  //         resolve(stdout);
  //       },
  //     );
  
  //     workerProcess.on('exit', (code) => {
  //       console.log(`子进程已退出，退出码 ${code}`);
  //       resolve(code);
  //     });
  //   })
  // }
}
