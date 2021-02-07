<template>
  <div class="compression_page">
    <h1 class="title">Compressor</h1>
    <div class="comment">Upload facial image to compress</div>
    <div class="container">
      <div class="image_list" v-if="isAdded">
        <div class="image_card" v-for="(item, i) in fileData" :key="i">
          <div class="image_wrapper">
            <img class="image" :src="item.data" />
          </div>
          <div class="image_name">{{ item.name }}</div>
        </div>
      </div>
      <div
        class="upload_area clickable"
        :style="{ paddingTop: `${imageListHeight}px` }"
        ref="uploadArea"
        @click="selectFile"
      >
        <div class="tips" v-if="!isAdded">
          <div class="tips_line">Click here to select image to upload</div>
          <div class="tips_line">or drag & drop image here 😊</div>
        </div>
        <div class="tips" v-else>Add more...</div>
        <input
          class="none"
          type="file"
          accept="image/*"
          multiple
          @change="getFile"
          ref="fileInput"
        />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      fileData: [],
    };
  },
  computed: {
    isAdded() {
      return this.fileData.length > 0;
    },
    imageListHeight() {
      return Math.min(350, this.fileData.length * 100);
    },
  },
  mounted() {
    const dropArea = this.$refs.uploadArea;
    dropArea.addEventListener('drop', this.dropEvent);
    dropArea.addEventListener('dragenter', this.preventFn);
    dropArea.addEventListener('dragover', this.preventFn);
  },
  methods: {
    selectFile() {
      this.$refs.fileInput.click();
    },
    addFiles(files) {
      console.log(files);
      files.forEach((file) => {
        const fileReader = new FileReader();
        fileReader.addEventListener('load', () => {
          this.fileData.push({
            name: file.name,
            data: fileReader.result,
          });
          console.log(this.fileData);
        });
        fileReader.readAsDataURL(file);
      });
    },
    getFile(e) {
      this.addFiles(e.target.files);
    },
    dropEvent(e) {
      this.preventFn(e);
      this.addFiles(e.dataTransfer.files);
    },
    preventFn(e) {
      e.stopPropagation();
      e.preventDefault();
    },
  },
};
</script>

<style lang="scss" scoped>
.compression_page {
  margin: 0 auto;
  padding-top: 60px;
  width: 900px;

  .title {
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }

  .container {
    $borderRadius: 4px;

    height: 450px;
    width: 100%;
    position: relative;

    .upload_area {
      position: absolute;
      top: 0;
      left: 0;
      z-index: 0;

      height: 100%;
      width: 100%;

      display: flex;
      justify-content: center;
      align-items: center;

      border: var(--standard-border);
      border-radius: $borderRadius;

      .tips {
        font-size: 32px;
        opacity: 0.3;

        .tips_line {
          margin-bottom: 12px;
        }
      }
    }

    .image_list {
      position: absolute;
      left: 0;
      top: 0;
      z-index: 1;

      width: 100%;
      max-height: 350px;
      overflow-y: auto;
      padding: 0 15px;

      display: flex;
      flex-direction: column;

      background-color: #fff;
      border-radius: $borderRadius;
      box-shadow: 2px 4px 24px var(--shadow);

      .image_card {
        height: 100px;
        padding: 15px 0;
        display: flex;

        border-bottom: var(--standard-border);

        &:last-child {
          border-bottom: none;
        }

        .image_wrapper {
          height: 70px;
          width: 70px;
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

        .image_name {
          color: var(--text2);
        }
      }
    }
  }
}
</style>
