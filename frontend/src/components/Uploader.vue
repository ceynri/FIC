<template>
  <div class="uploader">
    <div class="container_border"></div>
    <overlay-scrollbars class="image_list" v-viewer="viewerOptions" ref="imageList">
      <div class="image_item" v-for="(item, i) in fileData" :key="item.name">
        <div class="image_wrapper">
          <img class="image" :src="item.data" />
        </div>
        <div class="image_info">
          <div class="image_name">
            {{ item.name }}
          </div>
          <div class="image_size">
            {{ getSize(item.size) }}
          </div>
        </div>
        <div class="btn_wrapper">
          <button class="btn clickable" @click="watchOriginalImage(i)">
            <IconBase width="20px" height="20px" name="zoom in">
              <ZoomInIcon />
            </IconBase>
          </button>
          <button class="btn clickable" @click="deleteFile(i)">
            <IconBase width="20px" height="20px" name="cancel">
              <CancelIcon />
            </IconBase>
          </button>
        </div>
      </div>
    </overlay-scrollbars>
    <form class="upload_area clickable" ref="uploadArea" @click="selectFiles">
      <IconBase
        class="add"
        :width="isAdded ? 80 : 160"
        :height="isAdded ? 80 : 160"
        name="add"
        v-if="dragOver"
      >
        <AddIcon />
      </IconBase>
      <div class="tips" v-else-if="!isAdded">
        <div class="tips_line">Click here to select image to upload</div>
        <div class="tips_line">or drag & drop image here 😊</div>
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
import ZoomInIcon from '@/components/icons/ZoomInIcon.vue';
import CancelIcon from '@/components/icons/CancelIcon.vue';
import AddIcon from '@/components/icons/AddIcon.vue';

const BTYE_PER_KB = 2 ** 10;
const BTYE_PER_MB = 2 ** 20;

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
  },
  data() {
    return {
      fileData: [],
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
      return this.fileData.length > 0;
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
    selectFiles() {
      this.$refs.fileInput.click();
    },
    /**
     * 从原生Input文件中获取files
     */
    getFiles(e) {
      this.addFiles(e.target.files);
    },
    /**
     * 将文件保存为变量
     */
    addFiles(files) {
      console.debug('add files:', files);
      files.forEach((file) => {
        // 去重
        if (this.fileNameSet.has(file.name)) {
          // TODO 可改为 Bubbling prompt
          console.warn(`存在重复的文件：${file.name}`);
          // alert(`存在重复的文件：${file.name}`);
          return;
        }
        this.fileNameSet.add(file.name);
        // 读取文件并保存
        const fileReader = new FileReader();
        fileReader.addEventListener('load', () => {
          this.fileData.push({
            name: file.name,
            data: fileReader.result,
            size: file.size,
          });
          console.debug('fileData:', this.fileData);
        });
        fileReader.readAsDataURL(file);
      });
    },
    /**
     * 删除特定的文件
     */
    deleteFile(i) {
      this.fileNameSet.delete(this.fileData[i].name);
      this.fileData.splice(i, 1);
    },
    /**
     * 拖拽上传文件的事件处理
     */
    dropEvent(e) {
      this.preventFn(e);
      this.addFiles(e.dataTransfer.files);
    },
    /**
     * 禁止原生事件避免触发打开图片的原生行为
     */
    preventFn(e) {
      e.stopPropagation();
      e.preventDefault();
    },
    getSize(btye, remainder = 2) {
      if (btye < BTYE_PER_KB) {
        return `${btye} B`;
      }
      if (btye < BTYE_PER_MB) {
        return `${(btye / BTYE_PER_KB).toFixed(remainder)} KB`;
      }
      return `${(btye / BTYE_PER_MB).toFixed(remainder)} MB`;
    },
    watchOriginalImage(i) {
      if (!this.viewer) {
        console.warn('viewer未初始化');
        return;
      }
      this.viewer.view(i);
    },
  },
  components: {
    ZoomInIcon,
    CancelIcon,
    AddIcon,
  },
};
</script>

<style lang="scss" scoped>
.uploader {
  $borderRadius: 8px;

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
    border-radius: $borderRadius;
    transition: border var(--duration);
  }

  &:hover {
    .container_border {
      border-color: var(--border2);
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

    background-color: #fff;
    border-radius: $borderRadius;
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
        height: $contentHeight;
        width: $contentHeight * 2;
        flex: none;

        display: flex;
        justify-content: space-evenly;
        align-items: center;

        .btn {
          opacity: 0.3;
          transition: opacity var(--duration);

          &:hover {
            opacity: 1;
          }
        }
      }
    }
  }
}
</style>
