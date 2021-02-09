<template>
  <section class="demo page_frame">
    <header class="title_wrapper">
      <h1 class="title">Demo</h1>
      <div class="comment">Try to upload a facial image</div>
    </header>
    <Uploader v-if="!image" accept="image/*" :multiple="false" @uploaded="getImage" />
    <div v-else class="container">
      <div class="preview_panel card">
        <div class="image_wrapper">
          <img class="image" :src="image.data" :alt="image.name" ref="image" />
        </div>
        <div class="image_info">
          <div class="image_name">
            {{ image.name }}
          </div>
          <div class="image_size">{{ this.image.width }} × {{ this.image.height }}</div>
          <div class="image_size">{{ sizeFormat(image.size) }}</div>
        </div>
      </div>
      <div class="setting_panel card">
        <div class="setting_item">ratio</div>
        <div class="setting_item">quality</div>
        <div class="setting_item">xxx...</div>
        <button class="next_btn clickable">
          <div class="text">Next</div>
          <IconBase width="20px" height="20px" icon-name="next">
            <RightArrowIcon />
          </IconBase>
        </button>
      </div>
    </div>
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';
import RightArrowIcon from '@/components/icons/RightArrowIcon.vue';

export default {
  data() {
    return {
      image: null,
    };
  },
  methods: {
    getImage(file) {
      this.image = file;
    },
  },
  components: {
    Uploader,
    RightArrowIcon,
  },
};
</script>

<style lang="scss" scoped>
.demo {
  .card {
    border-radius: 8px;
    background-color: var(--bg2);
    box-shadow: 4px 8px 72px var(--shadow);

    transition: box-shadow 0.5s ease;

    &:hover {
      box-shadow: 4px 8px 96px 16px var(--shadow);
    }
  }

  .container {
    margin: 30px 0;
    display: flex;
    align-items: flex-start;

    .preview_panel {
      width: 300px;
      margin-right: 40px;

      .image_wrapper {
        width: 100%;
        max-height: 600px;
        overflow: hidden;

        .image {
          border-radius: 8px;
          width: 100%;
          height: auto;
        }
      }

      .image_info {
        color: var(--text2);
        margin: 12px 16px 16px;

        .image_name,
        .image_size {
          @include no-wrap;
        }
        .image_name {
          color: var(--text);
          font-size: 16px;
          margin-bottom: 10px;
        }

        .image_size {
          font-size: 12px;
          margin-top: 8px;
        }
      }
    }

    .setting_panel {
      flex: 1;
      padding: 40px;

      display: flex;
      flex-direction: column;
      // justify-content: space-evenly;

      position: relative;

      .setting_item {
        font-size: 18px;
        margin-bottom: 40px;
      }
      .next_btn {
        position: absolute;
        right: 40px;
        bottom: 30px;

        display: flex;
        align-items: center;

        color: var(--primary);
        border-radius: 100px;
        padding: 15px 20px;

        transition: box-shadow var(--duration);

        &:hover {
          box-shadow: 2px 4px 32px -4px var(--shadow);
        }

        .text {
          margin: 0 6px -0.1em;
          font-size: 20px;
        }
      }
    }
  }
}
</style>
