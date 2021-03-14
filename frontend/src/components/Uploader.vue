<template>
  <div class="uploader">
    <div class="container_border"></div>
    <overlay-scrollbars class="image_list" v-viewer="viewerOptions" ref="imageList">
      <div class="image_item" v-for="(item, i) in value" :key="item.name">
        <div class="image_wrapper">
          <img class="image" v-if="isImage(item.dataUrl)" :src="item.dataUrl" />
          <IconBase v-else width="60px" height="60px" icon-name="archive">
            <ArchiveIcon />
          </IconBase>
        </div>
        <div class="image_info">
          <div class="image_name">
            {{ item.name }}
          </div>
          <div class="image_size">
            {{ sizeFormat(item.size) }}
          </div>
        </div>
        <div class="btn_wrapper">
          <button v-if="item.result" class="btn clickable" @click="downloadResult(item.result)">
            <IconBase width="20px" height="20px" icon-name="download">
              <DownloadIcon />
            </IconBase>
          </button>
          <button v-if="isImage(item.dataUrl)" class="btn clickable" @click="viewImage(i)">
            <IconBase width="20px" height="20px" icon-name="zoom in">
              <ZoomInIcon />
            </IconBase>
          </button>
          <button class="btn clickable" @click="deleteFile(i)">
            <IconBase width="20px" height="20px" icon-name="cancel">
              <CancelIcon />
            </IconBase>
          </button>
        </div>
      </div>
    </overlay-scrollbars>
    <form class="upload_area clickable" ref="uploadArea" @click="clickToUpload">
      <IconBase
        class="add"
        :width="isAdded ? 80 : 160"
        :height="isAdded ? 80 : 160"
        icon-name="add"
        v-if="dragOver"
      >
        <AddIcon />
      </IconBase>
      <div class="tips" v-else-if="!isAdded">
        <div class="tips_line">Click here to upload image</div>
        <!-- TODO 判断是否为移动端 -->
        <div class="tips_line" v-if="true">or drop image here 😊</div>
      </div>
      <div class="tips" v-else>Add more...</div>
      <input
        class="none"
        type="file"
        :accept="accept"
        :multiple="multiple"
        @change="getFiles"
        ref="fileInput"
      />
    </form>
  </div>
</template>

<script>
import ArchiveIcon from '@/components/icons/ArchiveIcon.vue';
import DownloadIcon from '@/components/icons/DownloadIcon.vue';
import ZoomInIcon from '@/components/icons/ZoomInIcon.vue';
import CancelIcon from '@/components/icons/CancelIcon.vue';
import AddIcon from '@/components/icons/AddIcon.vue';

import readFile from '@/utils/readFile';
import { uploads } from '@/service';

export default {
  props: {
    accept: {
      type: String,
      default: '*',
    },
    multiple: {
      type: Boolean,
      default: true,
    },
    value: {
      type: Array,
      default: () => [],
    },
  },
  data() {
    return {
      fileNameSet: new Set(),
      dragOver: false,
      viewer: null,
      viewerOptions: {
        transition: false,
        toolbar: false,
        title: false,
        navbar: false,
      },
    };
  },
  computed: {
    isAdded() {
      return this.value.length > 0;
    },
  },
  mounted() {
    this.bindDragEvent();
    const imageListElem = this.$el.querySelector('.image_list');
    imageListElem.addEventListener('ready', () => {
      this.viewer = imageListElem.$viewer;
    });
  },
  methods: {
    /**
     * 监听uploadArea的拖拽事件
     */
    bindDragEvent() {
      const dropArea = this.$refs.uploadArea;
      if (!dropArea) {
        return;
      }
      dropArea.addEventListener('dragenter', () => {
        this.dragOver = true;
      });
      dropArea.addEventListener('dragover', (e) => {
        this.preventFn(e);
      });
      dropArea.addEventListener('dragleave', () => {
        this.dragOver = false;
      });
      dropArea.addEventListener('drop', (e) => {
        this.dragOver = false;
        this.dropEvent(e);
      });
    },
    /**
     * 封装fileInput的点击事件
     */
    clickToUpload() {
      this.$refs.fileInput.click();
    },
    /**
     * 拖拽上传文件的事件处理
     */
    dropEvent(e) {
      this.preventFn(e);
      const files = e.dataTransfer?.files;
      if (!this.multiple && files.length > 1) {
        alert('Please upload only one image');
        return;
      }
      this.addFiles(files);
    },
    /**
     * 从原生Input文件中获取files
     */
    getFiles(e) {
      this.addFiles(e.target.files);
    },
    /**
     * 将files保存为变量
     */
    addFiles(files) {
      console.debug('add files:', files);
      files.forEach(async (file) => {
        // 去重
        if (this.fileNameSet.has(file.name)) {
          // TODO 可改为 Bubbling prompt
          console.warn(`Duplicate file：${file.name}`);
          // alert(`Duplicate file：${file.name}`);
          return;
        }
        this.fileNameSet.add(file.name);

        // 读取文件的dataUrl与二进制格式
        try {
          const [dataUrl, blob] = await Promise.all([
            readFile(file, 'dataUrl'),
            readFile(file, 'blob'),
          ]);
          // 将图片保存为变量
          const fileItem = {
            dataUrl,
            blob,
            name: file.name,
            size: file.size,
            rawFile: file,
          };
          this.value.push(fileItem);

          console.debug('fileList:', this.value);
        } catch (e) {
          console.error(e);
          alert(`${file.name}上传失败，请重新上传`);
        }
      });
    },
    /**
     * 上传文件
     */
    async uploadFile(file) {
      try {
        const res = await uploads(file);
        console.log(res);
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * 删除特定的文件
     */
    deleteFile(i) {
      this.fileNameSet.delete(this.value[i].name);
      this.value.splice(i, 1);
    },
    /**
     * 查看大图
     */
    viewImage(i) {
      if (!this.viewer) {
        console.error('viewer is not exist!');
        return;
      }
      this.viewer.view(i);
    },
    /**
     * 判断 dataUrl 字符串是否为图像
     */
    isImage(dataUrl) {
      return dataUrl && dataUrl.startsWith('data:image');
    },
    /**
     * 禁止原生事件避免触发打开图片的原生行为
     */
    preventFn(e) {
      e.stopPropagation();
      e.preventDefault();
    },
  },
  components: {
    ArchiveIcon,
    DownloadIcon,
    ZoomInIcon,
    CancelIcon,
    AddIcon,
  },
};
</script>

<style lang="scss" scoped>
.uploader {
  min-height: 450px;
  width: 100%;
  display: flex;
  flex-direction: column;

  position: relative;

  .container_border {
    position: absolute;
    z-index: -1;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;

    border: var(--standard-border);
    border-radius: var(--border-radius);
    transition: all var(--duration);
  }

  &:hover {
    .container_border {
      border-color: var(--border2);
      background-color: rgba(0, 0, 0, 0.01);
    }

    .upload_area .tips {
      opacity: 0.4;
    }
  }

  .upload_area {
    width: 100%;
    flex: 1;
    min-height: 100px;

    display: flex;
    justify-content: center;
    align-items: center;

    .add {
      opacity: 0.3;
    }

    .tips {
      font-size: 32px;
      opacity: 0.3;
      transition: opacity var(--duration);

      .tips_line {
        margin-bottom: 0.5em;
        text-align: center;
      }
    }
  }

  .image_list {
    width: 100%;
    max-height: 550px;
    overflow-y: auto;
    padding: 0 15px;

    display: flex;
    flex-direction: column;

    background-color: var(--bg2);
    border-radius: var(--border-radius);
    box-shadow: 2px 4px 32px -4px var(--shadow);
    transition: box-shadow var(--duration);

    &:hover {
      box-shadow: 2px 4px 28px 4px var(--shadow);
    }

    .image_item {
      $height: 100px;
      $padding: 15px;
      $contentHeight: $height - 2 * $padding;

      height: $height;
      padding: $padding 0;
      display: flex;

      border-bottom: var(--standard-border);

      &:last-child {
        border-bottom: none;
      }

      .image_wrapper {
        height: $contentHeight;
        width: $contentHeight;
        flex: none;

        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 20px;

        .image {
          width: auto;
          height: auto;
          max-width: 100%;
          max-height: 100%;
        }
      }

      .image_info {
        color: var(--text2);
        min-width: 200px;
        flex: auto;
        padding: 0.5em 0;

        .image_name {
          margin-bottom: 0.5em;
          @include no-wrap;
        }
      }

      .btn_wrapper {
        $spaceDist: $contentHeight / 3;
        height: 100%;
        padding-left: $spaceDist;
        flex: none;

        display: flex;
        align-items: center;

        .btn {
          margin-right: $spaceDist;
          opacity: 0.3;
          transition: opacity var(--duration);
          color: var(--secondary);

          &:hover {
            opacity: 1;
          }
        }
      }
    }
  }
}
</style>
